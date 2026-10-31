# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import builtins
from typing import Literal, TypeAlias, cast, get_args, get_origin

import model_compressor.passes.adapt.base.fx.semantics as _semantics
import model_compressor.passes.adapt.base.nx.annotation as _annotation
import model_compressor.passes.adapt.base.nx.graph as _graph
import model_compressor.passes.annotate.base as annotate

Semantics: TypeAlias = _semantics.Semantics
"""NetworkX is the helper layer - the reference is still FX."""


_KEY_EDGE_SEMANTICS = "semantics"
"""The key under which an edge's semantics is stored."""


def has_semantics(graph: _graph.Graph, edge: _graph.Edge) -> bool:
    """Verify whether an edge has a semantics."""
    return _KEY_EDGE_SEMANTICS in graph.edges[edge]


def set_semantics(graph: _graph.Graph, edge: _graph.Edge, semantics: Semantics) -> None:
    """Write an edge's semantics."""
    graph.edges[edge][_KEY_EDGE_SEMANTICS] = semantics


def get_semantics(graph: _graph.Graph, edge: _graph.Edge) -> Semantics:
    """Read an edge's semantics."""
    return cast(Semantics, graph.edges[edge][_KEY_EDGE_SEMANTICS])


def _get_input_port_type(schema: annotate.SchemaProtocol, port: _graph.Port) -> annotate.DType:
    """Read the data type of an operator's input port.

    Args:
        schema: The (source or optimized) schema of an operator.
        port: The operator's input port.

    Returns:
        The type of the data read by the operator at the input port.

    """
    head, *path = port
    if len(path) == 0:
        (arg,) = tuple(arg for arg in schema.signature.input if (arg.name == head))
        type = arg.type
    else:
        (arg,) = tuple(arg for arg in schema.signature.input if (arg.name == head))
        assert get_origin(arg.type) is builtins.list
        (type,) = get_args(arg.type)
    assert get_origin(type) is Literal
    (type,) = get_args(type)
    return cast(annotate.DType, type)


def _get_output_port_type(schema: annotate.SchemaProtocol, port: _graph.Port) -> annotate.DType:
    """Read the data type of an operator's output port.

    Args:
        schema: The (source or optimized) schema of an operator.
        port: The operator's output port.

    Returns:
        The type of the data written by the operator to the output port.

    """
    head, *path = port
    if len(path) == 0:
        (arg,) = tuple(arg for arg in schema.signature.output if (arg.name == head))
        type = arg.type
    else:
        assert head.startswith("getitem")
        if len(schema.signature.output) == 1:
            (arg,) = schema.signature.output
            assert get_origin(arg.type) is builtins.list
            (type,) = get_args(arg.type)
        else:
            (idx,) = path
            arg = schema.signature.output[idx]
            type = arg.type
    assert get_origin(type) is Literal
    (type,) = get_args(type)
    return cast(annotate.DType, type)


def get_input_port_semantics(graph: _graph.Graph, node: _graph.Node, port: _graph.Port) -> Semantics:
    """Read the semantics of an operator at an input port.

    Args:
        graph: The graph to which the operator belongs.
        node: The node pointing to the operator.
        port: The operator's input port.

    Returns:
        The semantics of the operator at the input port.

    """
    annotation = _annotation.get_annotation(graph, node)
    source = _get_input_port_type(annotation.source, port)
    chosen = _get_input_port_type(annotation.chosen, port)
    semantics = Semantics(manifest=source, latent=chosen)
    return semantics


def get_output_port_semantics(graph: _graph.Graph, node: _graph.Node, port: _graph.Port) -> Semantics:
    """Read the semantics of an operator at an output port.

    Args:
        graph: The graph to which the operator belongs.
        node: The node pointing to the operator.
        port: The operator's output port.

    Returns:
        The semantics of the operator at the output port.

    """
    annotation = _annotation.get_annotation(graph, node)
    source = _get_output_port_type(annotation.source, port)
    chosen = _get_output_port_type(annotation.chosen, port)
    semantics = Semantics(manifest=source, latent=chosen)
    return semantics
