# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import copy

import networkx as nx

import model_compressor.passes.adapt.base as base


def _analyze_subgraph(subgraph: base.nx.Subgraph) -> base.nx.Subgraph:
    """Find the entrying edges, exiting edges, and dominator of a subgraph.

    Args:
        subgraph: A subgraph.

    Returns:
        The subgraph, whose entrying edges, exiting edges, and dominator have been found.

    """
    subgraph_ = nx.MultiDiGraph()
    subgraph_.add_edges_from(subgraph.edges)
    # find the entrying edges
    for node in subgraph_.nodes:
        if subgraph_.in_degree(node) == 0:
            outgoing = set(subgraph_.out_edges(node, keys=True))
            subgraph.entrying |= outgoing
    # find the exiting edges
    for node in subgraph_.nodes:
        if subgraph_.out_degree(node) == 0:
            incoming = set(subgraph_.in_edges(node, keys=True))
            subgraph.exiting |= incoming
    # find the dominator
    nodes = nx.topological_sort(subgraph_)
    if len(subgraph.entries) == 1:
        subgraph.dominators = copy.deepcopy(subgraph.entries)
    frontier = copy.deepcopy(subgraph.entrying)
    for node in nodes:
        incoming = set(subgraph_.in_edges(node, keys=True))
        outgoing = set(subgraph_.out_edges(node, keys=True))
        if incoming.issuperset(frontier):
            subgraph.dominators = {node}
            frontier = outgoing
        elif len(outgoing) > 0:
            frontier -= incoming
            frontier |= outgoing
        else:
            pass
    return subgraph


def induce_subgraphs(graph: base.nx.Graph) -> base.nx.Graph:
    """Create the subgraphs induced by the edge coloring.

    Args:
        graph: An edge-colored graph.

    Returns:
        The graph, whose color-induced subgraphs have been created.

    """
    semantics_to_subgraph: dict[base.nx.Semantics, base.nx.Subgraph] = {}
    for node in graph.nodes:
        for edge in graph.in_edges(node, keys=True):
            semantics = base.nx.get_semantics(graph, edge)
            subgraph = semantics_to_subgraph.get(semantics, base.nx.Subgraph(semantics))
            (pred, _, (_, _)) = edge
            subgraph.nodes |= {pred, node}
            subgraph.edges |= {edge}
            semantics_to_subgraph[semantics] = subgraph
    for subgraph in semantics_to_subgraph.values():
        subgraph = _analyze_subgraph(subgraph)
    for subgraph in semantics_to_subgraph.values():
        for edge in subgraph.edges:
            base.nx.set_subgraph(graph, edge, subgraph)
    return graph
