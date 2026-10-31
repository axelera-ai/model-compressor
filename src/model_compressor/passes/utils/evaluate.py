# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

"""Define the API to evaluate the functional operators in `GraphModule`s."""

import builtins
import dataclasses
import inspect
import operator
from collections.abc import Callable
from typing import Any, NamedTuple, TypeAlias, cast

import torch
import torch._ops
import torch.fx as fx
import torch.fx.immutable_collections

import model_compressor.ops.opset as opset
import model_compressor.passes.utils.rgetattr as rgetattr


class ArgumentValue(NamedTuple):
    """The value of an operator's input or output.

    Args:
        name: The name of the input/output.
        value: The value of the input/output.

    """

    name: str
    value: Any


@dataclasses.dataclass(frozen=True)
class ExecutionContext:
    """The objects required to evaluate a functional operator.

    Args:
        target: The functional operator pointed to by the node.
        input: The ordered list of name/value input pairs.
        output: For single-return operators, a single-item list name/value pair; for multi-return
            operators, a multi-item list of name/value pairs - one per return slot.

    """

    target: opset.Operator
    input: list[ArgumentValue]
    output: list[ArgumentValue]


Input: TypeAlias = dict[str, Any]
"""The input to a `GraphModule`."""


def evaluate_execution_contexts(graph_module: fx.GraphModule, input: Input) -> dict[fx.Node, ExecutionContext]:
    """Evaluate the input and output values for all `'call_function'` nodes in a `GraphModule`.

    !!! note

        This function assumes that the `GraphModule` has been normalized.

        ```python
        > graph_module = normalize.Normalize()(graph_module).graph_module
        ```

    Args:
        graph_module: A `GraphModule`.
        input: An input to the `GraphModule`.

    Returns:
        A mapping from nodes to execution contexts.

    Raises:
        ValueError: The input dictionary does not specify values for all `GraphModule.forward` parameters.

    """

    # check that the input is sufficiently specified
    placeholder_nodes = dict.fromkeys(graph_module.graph.find_nodes(op="placeholder"))
    expected = {node.name for node in placeholder_nodes}
    actual = set(input.keys())
    under = expected - actual
    if len(under) > 0:
        raise ValueError(f"Input must specify values for parameters {tuple(under)}")
    initial_env = {node: input[node.name] for node in placeholder_nodes}
    # evaluate the nodes
    interpreter = fx.Interpreter(graph_module, garbage_collect_values=False)
    with torch.no_grad():
        _ = interpreter.run(initial_env=initial_env)

    def _decode_v(v: Any) -> Any:
        match type(v):
            case builtins.list | torch.fx.immutable_collections.immutable_list:
                v = [_decode_v(item) for item in v]
            case fx.Node:
                v = interpreter.env[v]
            case _:
                v = v
        return v

    # index the results
    node_to_execution_context = {}
    for node, value in interpreter.env.items():
        if node.op == "call_function":
            assert callable(node.target)  # help MyPY
            if isinstance(node.target, torch._ops.OpOverload):
                input_ = [ArgumentValue(k, _decode_v(v)) for k, v in node.kwargs.items()]
                # assume that if a `torch._ops.OpOverload` does not return a `tuple`, its JIT schema declares a single output
                if not isinstance(value, tuple):
                    assert len(node.target._schema.returns) == 1, (
                        f"Node {node!r} declares more than one output but does not return a {tuple!r}"
                    )
                    value = (value,)
                output = [
                    ArgumentValue(argument.name, item)
                    for argument, item in zip(node.target._schema.returns, value, strict=True)
                ]
            else:
                if node.target is operator.getitem:
                    parameters = tuple(inspect.signature(node.target).parameters)
                    input_ = [ArgumentValue(name, _decode_v(v)) for name, v in zip(parameters, node.args, strict=True)]
                    output = [ArgumentValue("", value)]
                else:
                    raise ValueError(f"Node {node!r} has illegal target {node.target!r}")
            execution_context = ExecutionContext(target=node.target, input=input_, output=output)
            node_to_execution_context[node] = execution_context
    return node_to_execution_context


def _serialize_operator(op: opset.Operator) -> str:
    """Serialize an IR operator to a string.

    Args:
        op: An IR operator.

    Returns:
        The IR operator, in string form.

    """
    return op.__qualname__ if (op is operator.getitem) else str(op)


def _deserialize_operator(op: str) -> opset.Operator:
    """Deserialize a string into an IR operator.

    Args:
        op: An IR operator, in string form.

    Returns:
        The IR operator.

    """
    namespace = operator if (op == "getitem") else torch.ops
    return cast(Callable, rgetattr.rgetattr(namespace, op))


def serialize_execution_context(execution_context: ExecutionContext) -> dict[str, Any]:
    """Serialize an `ExecutionContext` into a dictionary.

    Args:
        execution_context: An `ExecutionContext`.

    Returns:
        The `ExecutionContext`, in dictionary form.

    """
    return {
        "target": _serialize_operator(execution_context.target),
        "input": execution_context.input,
        "output": execution_context.output,
    }


def deserialize_execution_context(execution_context: dict[str, Any]) -> ExecutionContext:
    """Deserialize a dictionary into an `ExecutionContext`.

    Args:
        execution_context: An `ExecutionContext`, in dictionary form.

    Returns:
        The `ExecutionContext`.

    """
    return ExecutionContext(
        target=_deserialize_operator(execution_context["target"]),
        input=execution_context["input"],
        output=execution_context["output"],
    )
