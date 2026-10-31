# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

"""Shared machinery to emit TensorRT-compilable ONNX from the adaptation's materialized
placement, contrasting STRICT (local) vs PERMISSIVE (non-local) assignment under the W picker.

Each materialized codec (encode/decode/recode) becomes a Q/DQ pair at its SI8 region's scale
(scales are invented -- only *equality* matters for TensorRT's reformat/fusion decisions); null
markers and ``fan_out`` are stripped as identities. STRICT keeps more codec round-trips than
PERMISSIVE, so its graph carries more Q/DQ boundaries -> more reformats -> higher latency.

A per-model ``emit.py`` supplies only its model factory (``() -> (GraphModule, kwargs)``); this
module does the rest and writes ``strict_W.onnx`` / ``permissive_W.onnx`` beside that ``emit.py``.
"""

import copy
import itertools
import os

import model_compressor.api as api
import model_compressor.passes as passes
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401 - registers torch.ops.quantized_decomposed

_QUANT = torch.ops.quantized_decomposed.quantize_per_tensor.default
_DEQUANT = torch.ops.quantized_decomposed.dequantize_per_tensor.default
_QMIN, _QMAX, _DT = -128, 127, torch.int8


def _is_codec(n):
    return n.op == "call_function" and any(k in str(n.target) for k in ("encode", "decode", "recode"))


def _is_strip(n):
    return n.op == "call_function" and (("mark_" in str(n.target)) or ("fan_out" in str(n.target)))


def _input_of(n):
    return n.kwargs["input"] if "input" in n.kwargs else n.args[0]


def build_materialized(factory, assignment, coalescence=passes.adapt.CoalescenceMode.WITH):
    """Materialize ``factory()``'s graph under the W picker, ``assignment``, and ``coalescence``."""
    gm, x = factory()
    c = api.generations.alpha.Config()
    c.annotation.input = x
    c.annotation.picker = api.generations.alpha.W_PICKER
    c.adaptation.assignment = assignment
    c.adaptation.coalescence = coalescence
    return api.prepare_before_optimization(gm, c), x


def _regions(gm):
    """Union-find over non-codec nodes via data edges (codec edges are cuts). Returns
    codec_node -> region-id (the SI8 region it bounds) and node -> region-id."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    def inputs(n):
        for a in itertools.chain(n.args, n.kwargs.values()):
            for item in a if isinstance(a, (list, tuple)) else [a]:
                if isinstance(item, torch.fx.Node):
                    yield item

    for n in gm.graph.nodes:
        if _is_codec(n):
            continue
        find(n)
        for p in inputs(n):
            if not _is_codec(p):
                union(p, n)
    # codec class = region of its interior-side neighbour (decode: producer; encode/recode: a consumer)
    codec_region = {}
    for n in gm.graph.nodes:
        if not _is_codec(n):
            continue
        if "decode" in str(n.target):
            codec_region[n] = find(_input_of(n))
        else:
            consumer = next((u for u in n.users if not _is_codec(u)), None)
            codec_region[n] = find(consumer) if consumer is not None else find(n)
    node_region = {n: find(n) for n in gm.graph.nodes if not _is_codec(n)}
    return codec_region, node_region


def _insert_qdq(graph, before, tensor, scale):
    with graph.inserting_before(before):
        q = graph.call_function(_QUANT, args=(tensor, scale, 0, _QMIN, _QMAX, _DT))
        dq = graph.call_function(_DEQUANT, args=(q, scale, 0, _QMIN, _QMAX, _DT))
    return dq


def lower(gm):
    """Faithfully lower the materialized placement: replace every codec with a Q/DQ pair at its
    region's scale, then strip null markers and ``fan_out``. No cat special-casing -- the
    strict/permissive contrast comes entirely from the materialized codec structure."""
    graph = gm.graph
    codec_region, node_region = _regions(gm)
    roots = sorted(set(codec_region.values()) | set(node_region.values()), key=id)
    region_scale = {r: 0.01 * i for i, r in enumerate(roots, start=1)}  # distinct scale per region
    for codec, region in list(codec_region.items()):
        dq = _insert_qdq(graph, codec, _input_of(codec), region_scale[region])
        codec.replace_all_uses_with(dq)
        graph.erase_node(codec)
    for n in [n for n in graph.nodes if _is_strip(n)]:
        n.replace_all_uses_with(_input_of(n))
        graph.erase_node(n)
    graph.lint()
    gm.recompile()
    return gm


def export(gm, x, path):
    import onnx

    with torch.no_grad():
        ep = torch.export.export(gm, args=(), kwargs=x)
    torch.onnx.export(ep, (), kwargs=x, f=path, dynamo=True)
    # embed external-data weights into a single self-contained file, then drop the sidecar
    model = onnx.load(path)
    onnx.save(model, path, save_as_external_data=False)
    if os.path.exists(path + ".data"):
        os.remove(path + ".data")


def main(factory, outdir):
    """Emit ``strict_W.onnx`` and ``permissive_W.onnx`` into ``outdir`` for ``factory``'s model."""
    for assignment, tag in [
        (passes.adapt.AssignmentMode.STRICT, "strict"),
        (passes.adapt.AssignmentMode.PERMISSIVE, "permissive"),
    ]:
        base, x = build_materialized(factory, assignment)
        gm = lower(copy.deepcopy(base))
        nq = sum(1 for n in gm.graph.nodes if n.target is _QUANT)
        path = os.path.join(outdir, f"{tag}_W.onnx")
        export(gm, x, path)
        print(f"{path}: {nq} Q/DQ pairs", flush=True)
