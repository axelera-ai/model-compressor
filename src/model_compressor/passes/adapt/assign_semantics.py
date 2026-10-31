# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import networkx as nx
import torch

import model_compressor.passes.adapt.base as base
import model_compressor.passes.adapt.commutation as commutation
import model_compressor.passes.adapt.markers  # noqa: F401 - register codec markers


def _assign_semantics_input(graph: base.nx.Graph, node: base.nx.Node) -> None:
    """Assign a semantics to all the edges outgoing from an input or parameter/buffer node.

    Args:
        graph: The graph to which the input or parameter/buffer belongs.
        node: The node pointing to the input or parameter/buffer.

    """
    edges = graph.out_edges(node, keys=True)
    all_port_semantics = {
        base.nx.get_input_port_semantics(graph, succ, input_port_succ) for (_, succ, (_, input_port_succ)) in edges
    }
    port_semantics, *_ = all_port_semantics
    assert all(semantics.is_compatible(port_semantics) for semantics in all_port_semantics)
    semantics = port_semantics
    for edge in edges:
        base.nx.set_semantics(graph, edge, semantics)


def _is_lossless_codec(graph: base.nx.Graph, node: base.nx.Node) -> bool:
    """Verify whether the codec marker pointed to by a node is lossless - and thus always commutes.

    Args:
        graph: The graph to which the codec marker belongs.
        node: The node pointing to the codec marker.

    Raises:
        ValueError: The node does not point to a codec marker.

    """
    target = base.nx.get_annotation(graph, node).target
    match target:
        case torch.ops.markers.mark_input_codec.int | torch.ops.markers.mark_input_codec.Tensor:
            port_semantics = base.nx.get_output_port_semantics(graph, node, ("",))
        case torch.ops.markers.mark_output_codec.int | torch.ops.markers.mark_output_codec.Tensor:
            port_semantics = base.nx.get_input_port_semantics(graph, node, ("input",))
        case _:
            raise ValueError(f"Node {node} must point to a codec marker, but has target {target}")
    return port_semantics.is_true


def _assign_semantics_mark_input_codec_strict(graph: base.nx.Graph, node: base.nx.Node) -> None:
    """Assign a semantics to all the edges outgoing from an input codec marker node.

    The semantics is that of the incoming edge if the codec is lossless.

    Args:
        graph: The graph to which the input codec marker belongs.
        node: The node pointing to the input codec marker.

    """
    edge, *_ = graph.in_edges(node, keys=True)
    edge_semantics = base.nx.get_semantics(graph, edge)
    port_semantics = base.nx.get_output_port_semantics(graph, node, ("",))
    if _is_lossless_codec(graph, node) and port_semantics.is_compatible(edge_semantics):
        semantics = edge_semantics
    else:
        semantics = port_semantics
    for edge in graph.out_edges(node, keys=True):
        base.nx.set_semantics(graph, edge, semantics)


def _assign_semantics_mark_input_codec_permissive(graph: base.nx.Graph, node: base.nx.Node) -> None:
    """Assign a semantics to all the edges outgoing from an input codec marker node.

    The semantics is that of the incoming edge if the codec is lossless or said semantics is
    compatible with that of the input codec marker's output port.

    Args:
        graph: The graph to which the input codec marker belongs.
        node: The node pointing to the input codec marker.

    """
    edge, *_ = graph.in_edges(node, keys=True)
    edge_semantics = base.nx.get_semantics(graph, edge)
    port_semantics = base.nx.get_output_port_semantics(graph, node, ("",))
    if port_semantics.is_compatible(edge_semantics):
        semantics = edge_semantics
    else:
        semantics = port_semantics
    for edge in graph.out_edges(node, keys=True):
        base.nx.set_semantics(graph, edge, semantics)


def _assign_semantics_mark_output_codec_strict(graph: base.nx.Graph, node: base.nx.Node) -> None:
    """Assign a semantics to all the edges outgoing from an output codec marker node.

    The semantics is that of the incoming edge if the codec is lossless.

    Args:
        graph: The graph to which the output codec marker belongs.
        node: The node pointing to the output codec marker.

    """
    edge, *_ = graph.in_edges(node, keys=True)
    edge_semantics = base.nx.get_semantics(graph, edge)
    port_semantics = base.nx.get_output_port_semantics(graph, node, ("",))
    if port_semantics.is_compatible(edge_semantics):
        semantics = edge_semantics
    else:
        semantics = port_semantics
    for edge in graph.out_edges(node, keys=True):
        base.nx.set_semantics(graph, edge, semantics)


def _assign_semantics_mark_output_codec_permissive(graph: base.nx.Graph, node: base.nx.Node) -> None:
    """Assign a semantics to all the edges outgoing from an output codec marker node.

    The semantics is always that of the incoming edge (i.e., unconditionally).

    Args:
        graph: The graph to which the output codec marker belongs.
        node: The node pointing to the output codec marker.

    """
    edge, *_ = graph.in_edges(node, keys=True)
    edge_semantics = base.nx.get_semantics(graph, edge)
    semantics = edge_semantics
    for edge in graph.out_edges(node, keys=True):
        base.nx.set_semantics(graph, edge, semantics)


def _assign_semantics_non_marker(graph: base.nx.Graph, node: base.nx.Node) -> None:
    """Assign a semantics to all the edges outgoing from a non-marker operator node.

    For each block of shared-semantics edges, the semantics is that of the incoming edges if it is
    unique, or a brand new semantics if it's not (in which case, the incoming edges are recolored
    with said brand new semantics).

    Args:
        graph: The graph to which the non-marker operator belongs.
        node: The node pointing to the non-marker operator.

    """
    for edge_block in commutation.compute_edge_blocks(graph, node):
        all_edge_semantics = {base.nx.get_semantics(graph, edge) for edge in edge_block.incoming}
        all_port_semantics = {
            base.nx.get_input_port_semantics(graph, node, input_port) for (_, _, (_, input_port)) in edge_block.incoming
        } | {
            base.nx.get_output_port_semantics(graph, node, output_port)
            for (_, _, (output_port, _)) in edge_block.outgoing
        }
        semantics = all_edge_semantics.pop() if (len(all_edge_semantics) == 1) else all_port_semantics.pop()
        for edge in edge_block.incoming | edge_block.outgoing:
            base.nx.set_semantics(graph, edge, semantics)


def assign_semantics_strict(graph: base.nx.Graph) -> base.nx.Graph:
    """Assign a semantics to every edge in a graph.

    Args:
        graph: A graph.

    Returns:
        The graph, to all edges of which a semantics has been assigned.

    Raises:
        ValueError: A node has opcode ``"call_method"`` or ``"call_module"``.

    """
    # Invariant: every edge marked as true has the same true semantics at both ends.
    # Invariant: every edge marked as fake has the same fake semantics at both ends.
    for node in nx.topological_sort(graph):
        annotation = base.nx.get_annotation(graph, node)
        match annotation.op:
            case "placeholder" | "get_attr":
                _assign_semantics_input(graph, node)
            case "output":
                pass
            case "call_function":
                match annotation.target:
                    case torch.ops.markers.mark_input_codec.int | torch.ops.markers.mark_input_codec.Tensor:
                        _assign_semantics_mark_input_codec_strict(graph, node)
                    case torch.ops.markers.mark_output_codec.int | torch.ops.markers.mark_output_codec.Tensor:
                        _assign_semantics_mark_output_codec_strict(graph, node)
                    case _:
                        _assign_semantics_non_marker(graph, node)
            case _:
                raise ValueError(f"Node {node} has illegal opcode {annotation.op}")
    return graph


def assign_semantics_permissive(graph: base.nx.Graph) -> base.nx.Graph:
    """Assign a semantics to every edge in a graph.

    Args:
        graph: A graph.

    Returns:
        The graph, to all edges of which a semantics has been assigned.

    Raises:
        ValueError: A node has opcode ``"call_method"`` or ``"call_module"``.

    """
    # Invariant: every edge marked as true has the same true semantics at both ends.
    # Asymmetry: an edge marked as fake _can_ have the same true semantics at both ends; this is
    # because codec markers _can_ be transparent to fake semantics.
    for node in nx.topological_sort(graph):
        annotation = base.nx.get_annotation(graph, node)
        match annotation.op:
            case "placeholder" | "get_attr":
                _assign_semantics_input(graph, node)
            case "output":
                pass
            case "call_function":
                match annotation.target:
                    case torch.ops.markers.mark_input_codec.int | torch.ops.markers.mark_input_codec.Tensor:
                        _assign_semantics_mark_input_codec_permissive(graph, node)
                    case torch.ops.markers.mark_output_codec.int | torch.ops.markers.mark_output_codec.Tensor:
                        _assign_semantics_mark_output_codec_permissive(graph, node)
                    case _:
                        _assign_semantics_non_marker(graph, node)
            case _:
                raise ValueError(f"Node {node} has illegal opcode {annotation.op}")
    return graph
