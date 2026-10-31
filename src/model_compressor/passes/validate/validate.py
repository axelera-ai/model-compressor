"""Define the API to validate that `GraphModule`s are in canonical form.

This module enforces one structural invariant of the demo flow: every functional node whose
runtime output is a *container* of tensors (a Python list or tuple) is unpacked exclusively via
``operator.getitem`` consumers, with each integer index in the container covered exactly once.
Under this invariant, every edge that any downstream pass needs to consider carries a single
tensor; the container values appear only as the immediate output of their producer and the
immediate input to their unpacking getitems.

Why the invariant exists
------------------------

Codec operators (``xaten.codec.default``, ``xaten.decode.default``, ``xaten.encode.default``,
``xaten.recode.default``) act per-tensor: they accept a single tensor and return a single tensor
with a different data type. They do not operate on lists or tuples.
Some operators in ``OPSET`` produce container values:

* single-return ``List[Tensor]`` operators - ``xaten.kv_cache.default``,
  ``aten.split_with_sizes.default``, ``aten.copy.t``;
* multi-return operators that FX represents as one tuple-valued node -
  ``aten.max.dim``, ``aten.min.dim``, ``aten.sort.*``, ``aten.topk.default``,
  ``aten.max_pool{2,3}d_with_indices.default``, ``aten.native_{layer,group}_norm.default``,
  ``aten.native_dropout.default``, ``aten._embedding_bag.default`` and the corresponding
  ``*_backward.default`` and ``_native_batch_norm_legit*`` ops.

If any consumer of such a node held the container directly, no cast could be spliced on that
edge. Forcing the unpacking via ``operator.getitem`` ensures every tensor-carrying edge is a
tensor-typed edge, which is what the cast-wrapping passes assume.

Design choices reflected here
-----------------------------

* **Structural, not dtype-aware.** The check depends only on graph topology and runtime values.
  It does *not* consult any schema annotation, candidate set, or picker decision. That keeps it
  cleanly separable from the cast-placement concerns, and lets passes downstream of it
  (``wrap_with_casts``, ``insert_casts``) treat the invariant as a precondition they need not
  re-verify.

* **Unconditional.** We validate the canonical form on every graph regardless of whether any
  container output will end up needing a cast. A conditional variant ("validate only when the
  chosen schema demands a different element dtype") would couple this pass to the picker, which
  is the coupling we are explicitly trying to avoid. The cost of unconditional validation is
  bounded: container-producing operators are sparse in ``OPSET``, and ``torch.export`` already
  unpacks multi-return values via ``operator.getitem``, so for typical exported graphs this
  validation is an assertion on existing structure rather than a request to insert nodes.

* **Bounded depth.** Across the entire ``OPSET`` (including training and backward operators),
  no operator produces a nested container - every container output is depth one. The validator
  therefore does not recurse: a single layer of unpacking is sufficient and complete.

"""

import operator

import torch.fx as fx

import model_compressor.passes.utils.evaluate as _evaluate


def validate(graph_module: fx.GraphModule, input: _evaluate.Input) -> None:
    """Verify the canonical-form invariant for every container-producing functional node.

    For each ``call_function`` node whose runtime output value is a list or tuple of tensors:

    * every user must be ``operator.getitem``;
    * each getitem must hold a literal integer index in ``args[1]``;
    * the set of indices used across all getitems must equal ``{0, ..., cardinality - 1}``,
      with no duplicates.

    The cardinality is read off the runtime evaluation, not off any schema annotation, so this
    function can run at any point in the flow as long as ``input`` is sufficient to evaluate
    the graph.

    Args:
        graph_module: A `GraphModule`.
        input: A mapping from names to values - one pair for each parameter of the
            `GraphModule.forward` method.

    Raises:
        ValueError: any container-producing node violates the invariant.

    """
    contexts = _evaluate.evaluate_execution_contexts(graph_module, input)
    for node, context in contexts.items():
        # Two "container producer" flavors:
        # - multi-return op (co-arity > 1): cardinality = len(context.output)
        # - single-return op whose runtime value is a list/tuple: cardinality = len(value)
        # In every other case the node is not a container producer; skip it.
        if len(context.output) > 1:
            cardinality = len(context.output)
        else:
            (only,) = context.output
            if not isinstance(only.value, (list, tuple)):
                continue
            cardinality = len(only.value)
        seen_indices: set[int] = set()
        for user in node.users:
            if user.target is not operator.getitem:
                raise ValueError(
                    f"validate_canonical: container-producing node {node.name!r} has "
                    f"non-getitem user {user.name!r}; unpack container outputs via "
                    "operator.getitem before any downstream pass"
                )
            if user.args[0] is not node:
                continue
            index = user.args[1]
            if not isinstance(index, int):
                raise ValueError(f"validate_canonical: getitem {user.name!r} has non-literal index {index!r}")
            if index in seen_indices:
                raise ValueError(
                    f"validate_canonical: container-producing node {node.name!r} has multiple "
                    f"getitems indexing position {index}"
                )
            seen_indices.add(index)
        missing = sorted(set(range(cardinality)) - seen_indices)
        if missing:
            raise ValueError(
                f"validate_canonical: container-producing node {node.name!r} produces a "
                f"length-{cardinality} container, but indices {missing} are not unpacked"
            )
