"""Define the API to validate operator-local assumptions about `OPSET` operators.

Where `validate.py` checks a *graph-global, topological* invariant - that every container-producing
node in a concrete `GraphModule` is unpacked exclusively via ``operator.getitem`` - this module
checks *operator-local, static* invariants: properties of a single operator's JIT schema, in
isolation, that hold independently of any graph, any sample input, or any execution.

The two are complementary. `validate.py` guarantees that a single layer of ``operator.getitem``
unpacking is *applied*; this module guarantees the *precondition that makes one layer sufficient* -
that no operator ever declares a container nested inside another container, so a container value can
only ever be one unpacking deep.

Bounded container depth
-----------------------

A *container* JIT type is `torch.ListType`, `torch.TupleType`, or `torch.DictType`. The
*container depth* of a JIT type is the maximum number of containers encountered along any path from
the type's root to a leaf:

    depth(leaf)            = 0                                   # scalar or tensor
    depth(Optional[t])     = depth(t)                           # transparent - see below
    depth(List[t])         = 1 + depth(t)
    depth(Tuple[t, ...])   = 1 + max(depth(t_i))
    depth(Dict[k, v])      = 1 + max(depth(k), depth(v))

A *flat* container has depth one (``aten.cat``'s ``Tensor[]`` input, ``aten.split_with_sizes``'s
``Tensor[]`` return); a *nested* (recursive) container has depth >= two (``Tensor[][]``, a list of
tuples, a tuple with a list-typed slot). The bounded-depth assumption is that **every** ``OPSET``
operator declares only flat containers - container depth at most one - across every input argument
and every return.

Design choices reflected here
-----------------------------

* **Static, not graph-aware.** The check consults only the operator's JIT schema
  (``OpOverload._schema``). It needs no `GraphModule`, no sample input, and no interpreter run,
  unlike `validate.py`. The validation unit is a single operator - the atom a node targets - which
  is why the public entry point takes one operator and a thin `validate_all` sweeps a set.

* **`Optional` is transparent.** ``Optional[t]`` has the depth of ``t``: it adds no container layer.
  So ``List[Optional[Tensor]]`` (``aten.index.Tensor``'s ``indices``) is flat - its elements are
  tensors-or-``None``, not containers. This mirrors the type grammar, which disjoins away
  optionals rather than treating them as a container.

* **Multi-return is an implicit tuple.** PyTorch models an N-tuple return as N separate ``returns``
  slots, each with its own type. A flat multi-return (``aten.max_pool2d_with_indices`` ->
  ``(Tensor, Tensor)``) is fine, but a slot that is *itself* a container (a hypothetical
  ``(Tensor, Tensor[])``) would require a second unpacking layer and is therefore a violation. We
  account for this by treating any return with more than one slot as wrapped in an implicit tuple.

* **`operator.getitem` is vacuously valid.** The builtin has no JIT schema, hence no declared
  container type to violate; it is accepted without inspection.

"""

from typing import cast

import torch
import torch._ops

import model_compressor.ops.opset as opset


def validate(op: opset.Operator) -> None:
    """Run every operator-local schema check against a single operator.

    Currently the suite consists of the bounded-container-depth check; further operator-local
    checks can be added here as the design's assumptions accrue.

    Args:
        op: An `OPSET` operator.

    Raises:
        ValueError: The operator violates an operator-local schema assumption.

    """
    _validate_bounded_depth(op)


def validate_all(operators: set[opset.Operator]) -> None:
    """Run `validate` against every operator in a set.

    Args:
        operators: A set of `OPSET` operators (e.g. ``opset.OPSET``).

    Raises:
        ValueError: Any operator violates an operator-local schema assumption.

    """
    for op in operators:
        validate(op)


def _validate_bounded_depth(op: opset.Operator) -> None:
    """Verify that an operator declares no nested container in its JIT schema.

    Every input argument and every (single) return must have container depth at most one; a
    multi-return operator (an implicit tuple) must additionally have no container-typed slot, since
    such a slot would make the return effectively depth two.

    `operator.getitem` has no JIT schema and is accepted without inspection.

    Args:
        op: An `OPSET` operator.

    Raises:
        ValueError: An input, a single return, or a multi-return slot is a nested container.

    """
    if not isinstance(op, torch._ops.OpOverload):
        # `operator.getitem` (the only non-`OpOverload` in `OPSET`) has no JIT schema to inspect.
        return
    schema = op._schema
    # inputs: each argument type must be flat (depth <= 1)
    for argument in schema.arguments:
        assert isinstance(argument.type, torch._C.Type)
        depth = _container_depth(argument.type)
        if depth >= 2:
            raise ValueError(
                f"{op}: input {argument.name!r} is a nested container "
                f"{_describe(argument.type)!r} (container depth {depth}); only flat (depth-one) "
                "containers are allowed"
            )
    # outputs: a single return must be flat; a multi-return (implicit tuple) must have no
    # container-typed slot, which would make the effective return depth two
    returns = schema.returns
    is_multi_return = len(returns) > 1
    for index, return_ in enumerate(returns):
        assert isinstance(return_.type, torch._C.Type)
        depth = _container_depth(return_.type)
        if is_multi_return and depth >= 1:
            raise ValueError(
                f"{op}: multi-return slot [{index}] is a container {_describe(return_.type)!r} "
                f"(container depth {depth}); a container-typed slot of an implicit-tuple return is "
                "a nested container"
            )
        if (not is_multi_return) and depth >= 2:
            raise ValueError(
                f"{op}: return [{index}] is a nested container {_describe(return_.type)!r} "
                f"(container depth {depth}); only flat (depth-one) containers are allowed"
            )


def _container_depth(jit_type: torch._C.Type) -> int:
    """Compute the maximum container-nesting depth along any path through a JIT type.

    `torch.OptionalType` is transparent (adds no layer); `torch.ListType`, `torch.TupleType`, and
    `torch.DictType` each add one layer; every other (leaf) type has depth zero.

    Args:
        jit_type: A JIT type expression.

    Returns:
        The maximum number of containers encountered along any root-to-leaf path.

    """
    if isinstance(jit_type, torch.OptionalType):
        return _container_depth(jit_type.getElementType())
    if isinstance(jit_type, torch.ListType):
        return 1 + _container_depth(cast(torch._C.Type, jit_type.getElementType()))
    if isinstance(jit_type, torch.TupleType):
        elements = cast(tuple[torch._C.Type, ...], jit_type.elements())
        return 1 + max((_container_depth(element) for element in elements), default=0)
    if isinstance(jit_type, torch.DictType):
        return 1 + max(
            _container_depth(cast(torch._C.Type, jit_type.getKeyType())),
            _container_depth(cast(torch._C.Type, jit_type.getValueType())),
        )
    return 0


def _describe(jit_type: torch._C.Type) -> str:
    """Render a JIT type as a readable string.

    Args:
        jit_type: A JIT type expression.

    Returns:
        The type's annotation string when available, else its ``str`` form.

    """
    return jit_type.annotation_str if hasattr(jit_type, "annotation_str") else str(jit_type)


__all__ = ["validate", "validate_all"]
