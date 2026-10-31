# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

"""Re-fusion passes that turn a directly-traced HuggingFace Llama graph into one that speaks the
demo's ``xaten`` vocabulary.

``torch.export`` decomposes the model's RoPE, RMSNorm and attention into primitive ATen ops; these
passes recover them:

- `refuse_rope`      : decomposed ``apply_rotary_pos_emb`` -> ``xaten.rotary_positional_embeddings``
- `refuse_rms_norm`  : decomposed ``LlamaRMSNorm``        -> ``aten.rms_norm``
- `inject_kv_cache`  : insert ``xaten.kv_cache`` (NoCache => identity) on the pre-GQA-repeat k/v

They run on the raw ``ep.module()`` (which still carries ``node.meta`` provenance); a re-export +
``import_exported_program`` afterwards normalizes the graph and repopulates ``meta``.

NB never run a global ``eliminate_dead_code`` on these graphs -- when RoPE tables are computed in
the graph they carry stateful vmap/autocast HOP plumbing that DCE wrongly prunes. These passes erase
only the exact region nodes they replace.
"""

import operator

import model_compressor.ops  # noqa: F401 - register the xaten custom operators
import torch
import torch.fx as fx

_MUL = torch.ops.aten.mul.Tensor
_ADD = torch.ops.aten.add.Tensor
_POW = torch.ops.aten.pow.Tensor_Scalar
_UNSQUEEZE = torch.ops.aten.unsqueeze.default
_CAT = torch.ops.aten.cat.default
_SLICE = torch.ops.aten.slice.Tensor
_SDPA = torch.ops.aten.scaled_dot_product_attention.default
_RMS = torch.ops.aten.rms_norm.default
_ALIAS = torch.ops.aten.alias.default
_VIEW = torch.ops.aten.view.default
_ROPE = torch.ops.xaten.rotary_positional_embeddings.default
_KV = torch.ops.xaten.kv_cache.default

_APPLY_ROTARY_LINE = "apply_rotary_pos_emb("

# reshape/repeat plumbing skipped when tracing back to the tensor actually cached
_GQA_SKIP = {
    torch.ops.aten.reshape.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.view.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.contiguous.default,
    torch.ops.aten._assert_tensor_metadata.default,
    torch.ops.aten.to.dtype,
}


def _is(n, target) -> bool:
    return getattr(n, "op", None) == "call_function" and n.target is target


def _trace_tail(n) -> str:
    st = n.meta.get("stack_trace")
    if not st:
        return ""
    lines = [ln.strip() for ln in st.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _mod_tail(n) -> tuple[str | None, str | None]:
    s = n.meta.get("nn_module_stack")
    if not s:
        return None, None
    path, typ = list(s.values())[-1]  # (module_path_str, module_type_str)
    return path, typ


def refuse_rope(graph_module: fx.GraphModule) -> int:
    """Rewrite each decomposed ``apply_rotary_pos_emb`` region into two ``xaten`` rope ops.

    Each region output is an ``add`` (``q_embed`` / ``k_embed``); it is self-describing: the
    cos-branch mul yields ``x``, the sin-branch mul's non-trig arg is the ``cat`` (rotate_half), and
    the two ``unsqueeze`` inputs are the full-width ``cos``/``sin`` (which we slice to head_dim//2).
    """
    graph = graph_module.graph
    adds = [n for n in graph.nodes if _is(n, _ADD) and _APPLY_ROTARY_LINE in _trace_tail(n)]
    slice_cache: dict[tuple[int, int], fx.Node] = {}
    rewritten = 0
    for add in adds:
        muls = add.args
        assert len(muls) == 2 and all(_is(m, _MUL) for m in muls), f"unexpected rope add args: {muls}"
        x = cos_src = sin_src = None
        for m in muls:
            a, b = m.args
            trig = a if _is(a, _UNSQUEEZE) else b
            other = b if trig is a else a
            assert _is(trig, _UNSQUEEZE), f"no unsqueeze trig arg in {m}"
            if _is(other, _CAT):  # sin branch: other is rotate_half(x)
                sin_src = trig.args[0]
            else:  # cos branch: other is x
                x, cos_src = other, trig.args[0]
        assert x is not None and cos_src is not None and sin_src is not None, "rope region parse failed"
        half = x.meta["val"].shape[-1] // 2

        def _slice_half(src, _half=half):
            key = (id(src), _half)
            if key not in slice_cache:
                with graph.inserting_before(add):
                    slice_cache[key] = graph.call_function(_SLICE, args=(src, -1, 0, _half))
            return slice_cache[key]

        with graph.inserting_before(add):
            cos_h, sin_h = _slice_half(cos_src), _slice_half(sin_src)
            rope = graph.call_function(_ROPE, args=(x, cos_h, sin_h))
        add.replace_all_uses_with(rope)
        rewritten += 1

    # erase only the apply_rotary region nodes (reverse-topo), never a global DCE
    for n in reversed([n for n in graph.nodes if _APPLY_ROTARY_LINE in _trace_tail(n)]):
        assert len(n.users) == 0, f"apply_rotary node still used after rewrite: {n}"
        graph.erase_node(n)
    graph.lint()
    graph_module.recompile()
    return rewritten


def refuse_rms_norm(graph_module: fx.GraphModule) -> int:
    """Rewrite each decomposed ``LlamaRMSNorm`` region into an ``aten.rms_norm``.

    RMSNorm is a module, so ``nn_module_stack`` delimits it. The output is the ``weight * x`` mul;
    the input is the ``pow`` argument. The internal fp32 input-cast is shared with the residual add,
    so only genuinely-dead region nodes are erased.
    """
    import collections

    graph = graph_module.graph
    regions: dict[str, list] = collections.defaultdict(list)
    for n in graph.nodes:
        if getattr(n, "op", None) == "call_function" and (_mod_tail(n)[1] or "").endswith("RMSNorm"):
            regions[_mod_tail(n)[0]].append(n)

    count = 0
    for path, region in regions.items():
        out = weight = None
        for n in region:
            if n.target is _MUL:
                ga = next((a for a in n.args if isinstance(a, fx.Node) and a.op == "get_attr"), None)
                if ga is not None:
                    out, weight = n, ga
        assert out is not None and weight is not None, f"{path}: no weight-mul found"
        pow_node = next(n for n in region if n.target is _POW)
        inp = pow_node.args[0]
        eps = None
        for n in region:
            if n.target is _ADD:
                eps = next((float(a) for a in n.args if isinstance(a, (int, float))), eps)
        assert eps is not None, f"{path}: could not find eps"
        normalized_shape = list(weight.meta["val"].shape)
        with graph.inserting_before(out):
            rms = graph.call_function(_RMS, args=(inp, normalized_shape, weight, eps))
        out.replace_all_uses_with(rms)
        count += 1

    for n in reversed(
        [
            n
            for n in graph.nodes
            if getattr(n, "op", None) == "call_function" and (_mod_tail(n)[1] or "").endswith("RMSNorm")
        ]
    ):
        if len(n.users) == 0:
            graph.erase_node(n)
    graph.lint()
    graph_module.recompile()
    return count


def _arg(node, name: str, position: int = 0):
    """Read a (possibly kwargs-normalized) argument of a node."""
    if name in node.kwargs:
        return node.kwargs[name]
    return node.args[position] if len(node.args) > position else None


def canonicalize_views(graph_module: fx.GraphModule) -> tuple[int, int]:
    """Massage HF-dialect view plumbing into the demo's vocabulary (run AFTER import).

    Two ops the direct HF trace produces are absent from the demo database:

    - ``aten.alias`` -- a pure identity (here aliasing the attention mask); dropped outright.
    - ``aten.unsqueeze`` -- a size-1-inserting view feeding the GQA repeat; rewritten to the
      equivalent ``aten.view`` (which the database covers), using the node's own output shape.

    Returns the ``(aliases_dropped, unsqueezes_rewritten)`` counts.
    """
    graph = graph_module.graph
    dropped = 0
    for node in list(graph.nodes):
        if _is(node, _ALIAS):
            node.replace_all_uses_with(_arg(node, "self"))
            graph.erase_node(node)
            dropped += 1
    rewritten = 0
    for node in list(graph.nodes):
        if _is(node, _UNSQUEEZE):
            src = _arg(node, "self")
            size = [int(d) for d in node.meta["val"].shape]
            with graph.inserting_before(node):
                view = graph.call_function(_VIEW, kwargs={"self": src, "size": size})
            view.meta["val"] = node.meta.get("val")
            node.replace_all_uses_with(view)
            graph.erase_node(node)
            rewritten += 1
    graph.lint()
    graph_module.recompile()
    return dropped, rewritten


def _trace_back(n):
    """Walk back through GQA repeat / reshape plumbing to the tensor actually cached."""
    while getattr(n, "op", None) == "call_function" and n.target in _GQA_SKIP:
        n = n.args[0]
    return n


def inject_kv_cache(graph_module: fx.GraphModule) -> int:
    """Insert ``xaten.kv_cache`` (NoCache => identity) before each SDPA, on the pre-GQA-repeat k/v."""
    graph = graph_module.graph
    sdpas = [n for n in graph.nodes if _is(n, _SDPA)]
    order = {n: i for i, n in enumerate(graph.nodes)}
    for layer_idx, sdpa in enumerate(sdpas):
        k_arg, v_arg = sdpa.args[1], sdpa.args[2]
        total_seq_len = int(k_arg.meta["val"].shape[2])
        key_pre, value_pre = _trace_back(k_arg), _trace_back(v_arg)
        key_users, value_users = list(key_pre.users), list(value_pre.users)
        anchor = key_pre if order[key_pre] > order[value_pre] else value_pre
        with graph.inserting_after(anchor):
            kv = graph.call_function(_KV, args=(key_pre, value_pre, layer_idx, total_seq_len))
        with graph.inserting_after(kv):
            nk = graph.call_function(operator.getitem, args=(kv, 0))
        with graph.inserting_after(nk):
            nv = graph.call_function(operator.getitem, args=(kv, 1))
        for u in key_users:
            u.replace_input_with(key_pre, nk)
        for u in value_users:
            u.replace_input_with(value_pre, nv)
    graph.lint()
    graph_module.recompile()
    return len(sdpas)
