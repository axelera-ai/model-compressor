# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

"""Define node schema candidacy utilities.

Given a fully functional, keyword-argument-normalized model graph and a schema database, these
utilities map each `OpOverload` node to a collection of schemas that represent optimized operator
versions that the node can be lowered to.

Concretely, this step is meant to be executed:

* after the call to `source_schemas`;
* before the call to `choose_schemas`.

"""

import builtins
import types
import typing
from collections.abc import Set
from typing import Any, Literal, cast

import torch._ops
import torch.fx as fx
import torch.fx.immutable_collections
import torch.nn as nn

import model_compressor.passes.annotate.base as base
import model_compressor.passes.utils.evaluate as evaluate


def candidate_schemas(
    graph_module: fx.GraphModule,
    input: evaluate.Input,
    database: Set[base.SchemaProtocol],
) -> dict[fx.Node, list[base.SchemaProtocol]]:
    """Map ``OpOverload`` functional operator nodes to candidate operator schemas.

    Non-``OpOverload`` functional nodes are deliberately excluded since they have no JIT schemas,
    and therefore no reference that MC schemas can be validated against. MC-schema-like instances
    for these functional nodes are provided upstream by `synthesize_bases`, and copied through by
    `choose_bases`.

    !!! note

        This function assumes that the `GraphModule` is purely functional and
        keyword-argument-normalized.

    !!! note

        A node with no compatible schema maps to an empty candidate list - it is the consumer
        code's responsibility to decide how to handle this case.

    Args:
        graph_module: A `GraphModule`.
        database: A database of operator schemas.

    Returns:
        An `OpOverload`-functional-node-to-candidate-schemas mapping.

    """
    # reorganize the database by target - and memoize
    target_to_schema: dict[torch._ops.OpOverload, list[base.SchemaProtocol]] = {}
    for schema in database:
        target_to_schema.setdefault(cast(torch._ops.OpOverload, schema.target), []).append(schema)
    # traverse the model graph
    node_to_context = evaluate.evaluate_execution_contexts(graph_module, input)
    node_to_bases: dict[torch.fx.Node, list[base.SchemaProtocol]] = {}
    for node, context in node_to_context.items():
        schemas = [
            schema
            for schema in target_to_schema.get(cast(torch._ops.OpOverload, node.target), [])
            if _is_context_compatible_with_schema(context, schema)
        ]
        node_to_bases[node] = schemas
    return node_to_bases


def _is_context_compatible_with_schema(context: evaluate.ExecutionContext, schema: base.SchemaProtocol) -> bool:
    """Verify whether the `ExecutionContext` of an `OpOverload` functional node is compatible with a schema.

    !!! note

        This function does not verify compatibility on the output's side - the schema itself is
        guaranteed to be compatible since it's validated against the operator's JIT schema.

    Args:
        node: An `OpOverload` functional node.
        schema: An operator schema.

    Returns:
        Whether the node is compatible with the schema.

    """
    is_viable = True
    name_to_value = {arg.name: arg.value for arg in context.input}
    for arg in schema.signature.input:
        value = name_to_value[arg.name]
        is_viable &= _is_value_compatible_with_mc_type(value, arg.type)
    return is_viable


def _is_value_compatible_with_mc_type(value: Any, mc_type: Any) -> bool:
    """Verify whether a Python value is compatible with an MC type, traversing both in lockstep.

    Args:
        value: A Python value.
        mc_type: An MC type.

    Returns:
        Whether the value is compatible with the MC type.

    """
    match type(value):
        case builtins.list | torch.fx.immutable_collections.immutable_list:
            if typing.get_origin(mc_type) is list:
                (argument,) = typing.get_args(mc_type)
                return all(_is_value_compatible_with_mc_type(item, argument) for item in value)
            return False
        case builtins.bool:
            if typing.get_origin(mc_type) is Literal:
                (argument,) = typing.get_args(mc_type)
                return argument is base.ScalarDType.BOOL
            return False
        case builtins.complex:
            if typing.get_origin(mc_type) is Literal:
                (argument,) = typing.get_args(mc_type)
                return (argument is base.ScalarDType.COMPLEX) or (argument is base.ScalarDType.NUMBER)
            return False
        case builtins.float:
            if typing.get_origin(mc_type) is Literal:
                (argument,) = typing.get_args(mc_type)
                return (argument is base.ScalarDType.FLOAT) or (argument is base.ScalarDType.NUMBER)
            return False
        case builtins.int:
            if typing.get_origin(mc_type) is Literal:
                (argument,) = typing.get_args(mc_type)
                return (argument is base.ScalarDType.INT) or (argument is base.ScalarDType.NUMBER)
            return False
        case builtins.str:
            if typing.get_origin(mc_type) is Literal:
                (argument,) = typing.get_args(mc_type)
                return argument is base.ScalarDType.STRING
            return False
        case nn.Parameter | torch.Tensor:
            if typing.get_origin(mc_type) is Literal:
                (argument,) = typing.get_args(mc_type)
                return isinstance(argument, base.TensorDType)
            return False
        case torch.device:
            if typing.get_origin(mc_type) is Literal:
                (argument,) = typing.get_args(mc_type)
                return argument is base.ScalarDType.DEVICE
            return False
        case torch.dtype | torch.layout | torch.memory_format:
            # `torch.dtype`, `torch.layout`, and `torch.memory_format` values map to
            # `torch.IntType` in PyTorch's JIT type grammar
            if typing.get_origin(mc_type) is Literal:
                (argument,) = typing.get_args(mc_type)
                return argument is base.ScalarDType.INT
            return False
        case types.NoneType:
            if typing.get_origin(mc_type) is Literal:
                (argument,) = typing.get_args(mc_type)
                return argument is base.ScalarDType.NULL
            return False
        case _:
            return False
