# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import collections

import model_compressor.passes.adapt.base as base
import model_compressor.passes.adapt.markers as markers
import model_compressor.passes.adapt.splice as splice
import model_compressor.passes.utils.state as _state


def _insert_mark_input_codecs(state: _state.State, graph: base.nx.Graph) -> base.nx.PassResult:
    """Insert input codec markers upstream of each functional operator node.

    Args:
        state: The index of booked and free node names.
        graph: The graph.

    Returns:
        A pair whose first item is the graph where input codec markers have been inserted, and
            whose second item is a flag indicating whether any input codec marker was inserted.

    """
    modified = False
    for node in list(graph.nodes):
        annotation = base.nx.get_annotation(graph, node)
        if (annotation.op == "call_function") and (annotation.target not in markers.MARKERS_CODEC):
            for edge in list(
                graph.in_edges(node, keys=True)
            ):  # create a list whose size does not change during splicing
                (pred, _, (_, _)) = edge
                if base.nx.get_annotation(graph, pred).target not in markers.MARKERS_INPUT_CODEC:
                    _ = splice.splice_mark_input_codec(state, graph, edge)
                    modified |= True
    return base.nx.PassResult(graph=graph, modified=modified)


def _insert_mark_output_codecs(state: _state.State, graph: base.nx.Graph) -> base.nx.PassResult:
    """Insert output codec markers downstream of each functional operator node.

    Args:
        state: The index of booked and free node names.
        graph: A graph.

    Returns:
        A pair whose first item is the graph where output codec markers have been insertd, and
            whose second item is a flag indicating whether any input codec marker was inserted.

    """
    modified = False
    for node in list(graph.nodes):
        annotation = base.nx.get_annotation(graph, node)
        if (annotation.op == "call_function") and (annotation.target not in markers.MARKERS_CODEC):
            output_port_to_edges = collections.defaultdict(list)
            for edge in graph.out_edges(node, keys=True):
                (_, _, (output_port, _)) = edge
                output_port_to_edges[output_port].append(edge)
            for edges in output_port_to_edges.values():
                (_, succ, (_, _)), *_ = edges
                if not (
                    (len(edges) == 1) and (base.nx.get_annotation(graph, succ).target in markers.MARKERS_OUTPUT_CODEC)
                ):
                    _ = splice.splice_mark_output_codec(state, graph, edges)
                    modified |= True
    return base.nx.PassResult(graph=graph, modified=modified)


def insert_markers(state: _state.State, graph: base.nx.Graph) -> base.nx.PassResult:
    """Insert I/O codec markers around each functional operator node.

    Args:
        state: The index of booked and free node names.
        graph: A graph.

    Returns:
        A pair whose first item is the graph where I/O codec markers have been inserted, and whose
            second item is a flag indicating whether any output codec marker was inserted.

    """
    modified = False
    pass_result = base.nx.PassResult(graph, modified)
    pass_result = _insert_mark_input_codecs(state, pass_result.graph)
    modified |= pass_result.modified
    pass_result = _insert_mark_output_codecs(state, pass_result.graph)
    modified |= pass_result.modified
    return base.nx.PassResult(graph=graph, modified=modified)
