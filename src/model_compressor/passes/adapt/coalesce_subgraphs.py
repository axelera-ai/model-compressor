# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import TypeAlias, cast

import networkx as nx

import model_compressor.passes.adapt.base as base
import model_compressor.passes.adapt.markers as markers

_MetaNode: TypeAlias = base.nx.Semantics
"""A subgraph spanned by same-semantics edges."""

_MetaEdge: TypeAlias = tuple[_MetaNode, _MetaNode]
"""The codec marker exits of an upstream subgraph that are entries of a downstream subgraph."""

_MetaGraph: TypeAlias = nx.DiGraph
"""A graph capturing the dependencies between subgraphs spanned by same-semantics edges."""


_KEY_META_NODE_SUBGRAPH: str = "subgraph"
"""The key under which a meta-node's subgraph is stored."""


def _set_subgraph(meta_graph: _MetaGraph, meta_node: _MetaNode, subgraph: base.nx.Subgraph) -> None:
    """Write a meta-node's subgraph."""
    meta_graph.nodes[meta_node][_KEY_META_NODE_SUBGRAPH] = subgraph


def _get_subgraph(meta_graph: _MetaGraph, meta_node: _MetaNode) -> base.nx.Subgraph:
    """Read a meta-node's subgraph."""
    return cast(base.nx.Subgraph, meta_graph.nodes[meta_node][_KEY_META_NODE_SUBGRAPH])


_KEY_META_EDGE_BOUNDARY: str = "boundary"
"""The key under which a meta-edge's boundary is stored."""


def _set_boundary(meta_graph: _MetaGraph, meta_edge: _MetaEdge, boundary: set[base.nx.Node]) -> None:
    """Write a meta-edge's boundary."""
    meta_graph.edges[meta_edge][_KEY_META_EDGE_BOUNDARY] = boundary


def _get_boundary(meta_graph: _MetaGraph, meta_edge: _MetaEdge) -> set[base.nx.Node]:
    """Read a meta-edge's boundary."""
    return cast(set[base.nx.Node], meta_graph.edges[meta_edge][_KEY_META_EDGE_BOUNDARY])


def create_meta_graph(graph: base.nx.Graph) -> _MetaGraph:
    """Capture the dependencies between same-semantics subgraphs.

    Args:
        graph: A graph that has been edge-partitioned into same-semantics subgraphs.

    Returns:
        The meta-graph capturing the dependencies between same-semantics subgraphs.

    """
    meta_graph = _MetaGraph()
    for node in nx.topological_sort(graph):
        for edge in graph.out_edges(node, keys=True):
            subgraph = base.nx.get_subgraph(graph, edge)
            meta_node = subgraph.semantics
            meta_graph.add_node(meta_node)
            _set_subgraph(meta_graph, meta_node, subgraph)
        if base.nx.get_annotation(graph, node).target in markers.MARKERS_CODEC:
            incoming, *_ = graph.in_edges(node, keys=True)
            for outgoing in graph.out_edges(node, keys=True):
                upstream_subgraph = base.nx.get_subgraph(graph, incoming)
                downstream_subgraph = base.nx.get_subgraph(graph, outgoing)
                if (upstream_subgraph.semantics is not downstream_subgraph.semantics) and (
                    (node in upstream_subgraph.exits) and (node in downstream_subgraph.entries)
                ):
                    meta_edge = (upstream_subgraph.semantics, downstream_subgraph.semantics)
                    if meta_edge not in meta_graph.edges:
                        meta_graph.add_edge(*meta_edge)
                        _set_boundary(meta_graph, meta_edge, set())
                    boundary = _get_boundary(meta_graph, meta_edge)
                    boundary.add(node)
    return meta_graph


def coalesce_subgraphs(graph: base.nx.Graph) -> base.nx.Graph:
    """Coalesce same-semantics subgraphs while preserving the absolute dominance property.

    Args:
        graph: A graph that has been edge-partitioned into same-semantics subgraphs.

    Returns:
        The graph where same-semantics subgraphs have been coalesced.

    """
    meta_graph = create_meta_graph(graph)
    for meta_node in reversed(list(nx.topological_sort(meta_graph))):
        upstream_subgraph = _get_subgraph(meta_graph, meta_node)
        merge_candidates: list[_MetaNode] = []
        for meta_edge in list(meta_graph.out_edges(meta_node)):
            _, meta_succ = meta_edge
            downstream_subgraph = _get_subgraph(meta_graph, meta_succ)
            boundary = _get_boundary(meta_graph, meta_edge)
            assert boundary.issubset(upstream_subgraph.exits)
            if upstream_subgraph.semantics.is_compatible(downstream_subgraph.semantics) and (
                len(boundary) == len(upstream_subgraph.exits)
            ):
                merge_candidates.append(meta_succ)
        if len(merge_candidates) == 1:
            meta_succ = next(iter(merge_candidates))
            downstream_subgraph = _get_subgraph(meta_graph, meta_succ)
            # extend the downstream subgraph
            downstream_subgraph.entrying -= {(u, v, (q, p)) for (u, v, (q, p)) in downstream_subgraph.entrying if (u in (upstream_subgraph.nodes - upstream_subgraph.entries))}
            downstream_subgraph.entrying |= upstream_subgraph.entrying - downstream_subgraph.entrying
            downstream_subgraph.nodes |= upstream_subgraph.nodes
            downstream_subgraph.edges |= upstream_subgraph.edges
            # recolor the edges of the upstream subgraph
            for edge in upstream_subgraph.edges:
                base.nx.set_semantics(graph, edge, downstream_subgraph.semantics)
                base.nx.set_subgraph(graph, edge, downstream_subgraph)
            # update the meta-graph
            for meta_edge_old in list(meta_graph.in_edges(meta_node)):
                meta_pred, _ = meta_edge_old
                meta_edge_new = (meta_pred, meta_succ)
                if meta_edge_new not in meta_graph.edges:
                    meta_graph.add_edge(meta_pred, meta_succ)
                    _set_boundary(meta_graph, meta_edge_new, set())
                _set_boundary(
                    meta_graph,
                    meta_edge_new,
                    _get_boundary(meta_graph, meta_edge_new) | _get_boundary(meta_graph, meta_edge_old),
                )
                meta_graph.remove_edge(meta_pred, meta_node)
            meta_graph.remove_edge(meta_node, meta_succ)
            meta_graph.remove_node(meta_node)
    return graph
