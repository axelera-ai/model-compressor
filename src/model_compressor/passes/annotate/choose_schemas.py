# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

"""Define node schema choice utilities.

Given a fully functional, keyword-argument-normalized model graph and a mapping from `OpOverload`
nodes to candidate schemas, these utilities disambiguate which optimized operator version each
node is lowered to - the choices are annotated in nodes' metadata.

For non-`OpOverload` functional nodes (`operator.getitem`), no schema database lookup is
performed; the chosen schema is *derived from the producer's chosen schema*. For
``getitem(producer, index)``, the derived schema's input ``"a"`` is the producer's chosen output
(a ``list[...]`` for canonical container producers), input ``"b"`` is the source's index type
(``Literal[ScalarDType.INT]``), and output ``""`` is the element type extracted from the
producer's chosen container. The engine is inherited from the producer.

This derivation is symmetric to the way `wrap_with_casts` treats `operator.getitem`: both
passes treat the getitem as belonging to its producer. The canonical-form invariant validated
by `validate_canonical` guarantees that the producer's chosen output is a depth-one container,
that ``args[0]`` is an `fx.Node`, and that ``args[1]`` is a literal integer.

Concretely, this step is meant to be executed:

* after the call to `candidate_schemas`;
* before adaptation.

"""

import builtins
import operator
import typing
from typing import Literal, cast

import torch._ops
import torch.fx as fx

import model_compressor.passes.annotate.base as base
import model_compressor.passes.annotate.picker as _picker


def choose_schemas(
    graph_module: fx.GraphModule,
    node_to_candidates: dict[fx.Node, list[base.SchemaProtocol]],
    picker: _picker.Picker,
) -> fx.GraphModule:
    """Assign a schema annotation to every functional node in a model graph.

    For each `OpOverload` functional node, the picker selects one schema from its candidates.
    For each `operator.getitem` node, the chosen schema is derived from the producer's chosen
    schema (the node referenced by ``args[0]``).

    !!! note

        The function modifies the graph's annotations (not its topology nor composition) in-place.

    Args:
        graph_module: A `GraphModule`.
        node_to_candidates: An `OpOverload`-functional-node-to-candidate-schemas mapping.
        picker: A callable that, given a node and its list of candidate schemas, returns one.

    """
    schema: base.SchemaProtocol
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            if isinstance(node.target, torch._ops.OpOverload):
                schema = picker(node, node_to_candidates[node])
            else:
                if node.target is operator.getitem:
                    pred, idx = node.args
                    schema = synthesize_schema_for_operator_getitem(
                        cast(base.SchemaProtocol, base.get_annotation(pred).chosen), idx
                    )
                else:
                    raise ValueError(f"Node {node!r} targets unsupported operator {node.target!r}")
            annotation = base.get_annotation(node)
            annotation.chosen = schema
    return graph_module


def synthesize_schema_for_operator_getitem(schema: base.SchemaProtocol, idx: int) -> base.Schema:
    """Derive an ``operator.getitem`` schema from its sequence argument's schema and its index argument.

    Args:
        schema: The schema of the ``operator.getitem`` sequence argument.
        idx: The ``operator.getitem`` index argument.

    Returns:
        A synthetic ``Schema`` whose input ``"a"`` mirrors the sequence argument's output, whose input ``"b"`` is an
        index type, whose output ``""`` is the element type extracted from the sequence argument's output, and whose
        engine is inherited from the sequence argument's schema.

    Raises:
        ValueError: The sequence argument's schema output is not a depth-one container.

    """
    output = schema.signature.output
    # Disambiguation rule (per the canonical-form invariant):
    # - co-arity == 1: the producer's return must be a ``list[X]``; ``operator.getitem`` unpacks an element and yields
    #   its type - ``X``;
    # - co-arity > 1: the producer is multi-return; the ``operator.getitem`` picks the slot pointed to by its index
    #   argument and yields that slot's type.
    match len(output):
        case 0:
            raise ValueError(f"Schema {schema!r} has no return slots")
        case 1:
            (only,) = output
            if typing.get_origin(only.type) is not builtins.list:
                raise ValueError(f"Schema {schema!r}'s does not return a container: {only.type!r}")
            (type_,) = typing.get_args(only.type)
        case _:
            type_ = output[idx].type
    return base.Schema(
        engine=schema.engine,
        target=operator.getitem,
        signature=base.Signature(
            input=[
                base.Argument("a", output),
                base.Argument("b", Literal[base.ScalarDType.INT]),
            ],
            output=[base.Argument("", type_)],
        ),
        metadata="",
    )
