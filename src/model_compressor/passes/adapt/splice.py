# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import Literal

import torch

import model_compressor.passes.adapt.base as base
import model_compressor.passes.adapt.markers  # noqa: F401 - register codec markers
import model_compressor.passes.annotate.base as annotate
import model_compressor.passes.utils.state as _state


def splice_mark_input_codec(state: _state.State, graph: base.nx.Graph, edge: base.nx.Edge) -> base.nx.Node:
    """Splice an input codec marker along an edge.

    Args:
        state: The index of booked and free node names.
        graph: The graph to which the edge to be spliced belongs.
        edge: The edge to be spliced.

    Returns:
        The node that has been spliced along the edge.

    Raises:
        ValueError: The input codec marker must operate on a value whose type is not supported.

    """
    (pred, succ, (output_port_pred, input_port_succ)) = edge
    # choose the ``mark_input_codec`` overload that fits the semantics' data type
    semantics = base.nx.get_input_port_semantics(graph, succ, input_port_succ)
    match semantics.manifest:
        case annotate.ScalarDType.INT:
            target = torch.ops.markers.mark_input_codec.int
        case annotate.TensorDType():
            target = torch.ops.markers.mark_input_codec.Tensor
        case _:
            raise ValueError(f"No input codec marker for values of type {semantics.manifest} is defined")
    # create the annotation for the ``mark_input_codec`` node
    annotation = base.nx.get_annotation(graph, succ)
    annotation = base.nx.Annotation(
        op="call_function",
        target=target,
        kwargs={},
        value=None,
        annotation=annotate.Annotation(
            source=annotate.Schema(
                engine=annotation.source.engine,
                target=target,
                signature=annotate.Signature(
                    input=[annotate.Argument(name="input", type=Literal[semantics.manifest])],
                    output=[annotate.Argument(name="", type=Literal[semantics.manifest])],
                ),
                metadata="",
            ),
            chosen=annotate.Schema(
                engine=annotation.chosen.engine,
                target=target,
                signature=annotate.Signature(
                    input=[annotate.Argument(name="input", type=Literal[semantics.manifest])],
                    output=[annotate.Argument(name="", type=Literal[semantics.latent])],
                ),
                metadata="",
            ),
        ),
    )
    # edit the graph
    node = state.book(target)
    graph.add_node(node)
    base.nx.set_annotation(graph, node, annotation)
    graph.add_edge(pred, node, key=(output_port_pred, ("input",)))
    graph.add_edge(node, succ, key=(("",), input_port_succ))
    graph.remove_edge(pred, succ, key=(output_port_pred, input_port_succ))
    return node


def splice_mark_output_codec(state: _state.State, graph: base.nx.Graph, edges: list[base.nx.Edge]) -> base.nx.Node:
    """Splice an output codec marker between the root and the leaves of a star subgraph.

    Args:
        state: The index of booked and free node names.
        graph: The graph to which the star subgraph to be spliced belongs.
        edges: The edges of the star subgraph.

    Returns:
        The node spliced between the root and the leaves of the star subgraph.

    Raises:
        ValueError: The output codec marker must operate on a value whose type is not supported.

    """
    (pred, _, (output_port_pred, _)), *_ = edges
    # choose the ``mark_output_codec`` overload that fits the semantics' data type
    semantics = base.nx.get_output_port_semantics(graph, pred, output_port_pred)
    match semantics.manifest:
        case annotate.ScalarDType.INT:
            target = torch.ops.markers.mark_output_codec.int
        case annotate.TensorDType():
            target = torch.ops.markers.mark_output_codec.Tensor
        case _:
            raise ValueError(f"No output codec marker for values of type {semantics.manifest} is defined")
    # create the annotation for the ``mark_output_codec`` node
    annotation = base.nx.get_annotation(graph, pred)
    annotation = base.nx.Annotation(
        op="call_function",
        target=target,
        kwargs={},
        value=None,
        annotation=annotate.Annotation(
            source=annotate.Schema(
                engine=annotation.source.engine,
                target=target,
                signature=annotate.Signature(
                    input=[annotate.Argument(name="input", type=Literal[semantics.manifest])],
                    output=[annotate.Argument(name="", type=Literal[semantics.manifest])],
                ),
                metadata="",
            ),
            chosen=annotate.Schema(
                engine=annotation.chosen.engine,
                target=target,
                signature=annotate.Signature(
                    input=[annotate.Argument(name="input", type=Literal[semantics.latent])],
                    output=[annotate.Argument(name="", type=Literal[semantics.manifest])],
                ),
                metadata="",
            ),
        ),
    )
    # edit the graph
    node = state.book(target)
    graph.add_node(node)
    base.nx.set_annotation(graph, node, annotation)
    graph.add_edge(pred, node, key=(output_port_pred, ("input",)))
    for edge in edges:
        (_, succ, (_, input_port_succ)) = edge
        graph.add_edge(node, succ, key=(("",), input_port_succ))
    for edge in edges:
        (_, succ, (output_port_pred, input_port_succ)) = edge
        graph.remove_edge(pred, succ, key=(output_port_pred, input_port_succ))
    return node
