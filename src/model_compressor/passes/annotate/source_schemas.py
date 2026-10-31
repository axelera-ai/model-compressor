# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

"""Define schema synthesis utilities.

Given a runnable model graph and a sample input, these utilities annotate every functional node
with a schema describing the MC types of the concrete values flowing through it during evaluation.
The MC types are read off the runtime tensors and Python scalars themselves; they do not depend on
any schema database nor on any prior annotation pass.

Concretely, this step is meant to be executed:

* after canonicalization;
* before the call to `candidate_schemas`.

"""

import builtins
import enum
import operator
import types
from collections.abc import Callable
from typing import Any, Literal, cast

import torch
import torch.fx as fx
import torch.fx.immutable_collections

import model_compressor.passes.annotate.base as base
import model_compressor.passes.utils.evaluate as evaluate


def source_schemas(graph_module: fx.GraphModule, input: dict[str, Any]) -> fx.GraphModule:
    """Annotate functional nodes with a source schema synthesized from concrete evaluation values.

    !!! note

        This function annotates both ``OpOverload`` and ``operator.getitem`` functional nodes; the
        MC types are derived from the concrete values of runtime objects themselves.

    !!! note

        The function modifies the graph's annotations (not its topology nor composition) in-place.

    Args:
        graph_module: A `GraphModule`.
        input: A mapping from names to values - one pair for each parameter of `GraphModule.forward`.

    Returns:
        The `GraphModule`.

    Raises:
        ValueError: A node targets an unsupported operator.

    """
    node_to_context = evaluate.evaluate_execution_contexts(graph_module, input)
    for node, context in node_to_context.items():
        if isinstance(node.target, torch._ops.OpOverload):
            signature = base.Signature(
                input=[
                    base.Argument(arg.name, _synthesize_mc_type_from_value(f"input[{arg.name!r}]", arg.value))
                    for arg in context.input
                ],
                output=[
                    base.Argument(arg.name, _synthesize_mc_type_from_value(f"output[{arg.name!r}]", arg.value))
                    for arg in context.output
                ],
            )
        else:
            if node.target is operator.getitem:
                sequence, index = cast(tuple[fx.Node, int], node.args)
                source = cast(base.SchemaProtocol, base.get_annotation(sequence).source)
                output = source.signature.output
                match type(output):
                    case builtins.list | torch.fx.immutable_collections.immutable_list:
                        index = 0  # must be homogeneous
                    case builtins.tuple:
                        index = index  # can be heterogeneous
                    case _:
                        raise TypeError
                signature = base.Signature(
                    input=[
                        base.Argument("a", output),
                        base.Argument("b", Literal[base.ScalarDType.INT]),
                    ],
                    output=[base.Argument("", output)],
                )
            else:
                raise ValueError(f"Node {node!r} targets unsupported operator {node.target!r}")
        annotation = base.get_annotation(node)
        annotation.source = base.Schema(
            engine=_Source.SOURCE,
            target=cast(Callable, node.target),
            signature=signature,
            metadata="",
        )
    return graph_module


_TORCH_DTYPE_TO_TENSOR_DTYPE: dict[torch.dtype, base.TensorDType] = {
    torch.float32: base.TensorDType.F32,
    torch.float16: base.TensorDType.F16,
    torch.int8: base.TensorDType.SI8,
    torch.int64: base.TensorDType.SI64,
}
"""Map ``torch.dtype`` values to MC ``TensorDType`` members."""


def _synthesize_mc_type_from_value(where: str, value: Any) -> Any:
    """Synthesize the MC type that describes a concrete Python value.

    The returned type expression is in *typing form*, mirroring the convention of ``Schema``'s
    signature: leaves are ``Literal[<member>]`` and containers are ``list[ MCType ]``.

    Args:
        where: The path to the value being synthesized (for error messages).
        value: A Python value observed during evaluation.

    Returns:
        The MC type expression that the value would be annotated with.

    Raises:
        ValueError: The value's type does not correspond to any MC type, or a sequence is empty
            (and thus has no inferrable element type), or a sequence's elements disagree on type.

    """
    match type(value):
        case builtins.list | torch.fx.immutable_collections.immutable_list:
            if len(value) == 0:
                raise ValueError(f"{where}: empty list has no inferrable MC element type")
            element_types = [_synthesize_mc_type_from_value(f"{where}[{i}]", item) for i, item in enumerate(value)]
            head, *rest = element_types
            if any(other != head for other in rest):
                raise ValueError(f"{where}: list items have heterogeneous MC types {element_types!r}")
            return list[head]  # type: ignore [valid-type]
        case builtins.tuple:
            if len(value) == 0:
                raise ValueError(f"{where}: empty tuple has no inferrable MC element type")
            element_types = [_synthesize_mc_type_from_value(f"{where}[{i}]", item) for i, item in enumerate(value)]
            return tuple[element_types]  # type: ignore [valid-type]
        case builtins.bool:
            return Literal[base.ScalarDType.BOOL]
        case builtins.complex:
            return Literal[base.ScalarDType.COMPLEX]
        case builtins.float:
            return Literal[base.ScalarDType.FLOAT]
        case builtins.int:
            return Literal[base.ScalarDType.INT]
        case builtins.str:
            return Literal[base.ScalarDType.STRING]
        case types.NoneType:
            return Literal[base.ScalarDType.NULL]
        case torch.device:
            return Literal[base.ScalarDType.DEVICE]
        case torch.dtype | torch.layout | torch.memory_format:
            return Literal[base.ScalarDType.INT]
        case torch.Tensor | torch.nn.parameter.Parameter:
            if value.dtype not in _TORCH_DTYPE_TO_TENSOR_DTYPE:
                raise ValueError(f"{where}: tensor dtype {value.dtype!r} has no MC TensorDType counterpart")
            return Literal[_TORCH_DTYPE_TO_TENSOR_DTYPE[value.dtype]]
        case _:
            raise ValueError(f"{where}: value of type {type(value)!r} has no MC type counterpart")


class _Source(base.ComputingEngine):
    """The computing engine of the source graph - the graph as it executes before any optimization."""

    SOURCE = enum.auto()
