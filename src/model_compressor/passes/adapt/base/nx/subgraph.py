# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import dataclasses
from typing import cast

import model_compressor.passes.adapt.base.nx.graph as _graph
import model_compressor.passes.adapt.base.nx.semantics as _semantics


@dataclasses.dataclass
class Subgraph:
    """A subgraph spanned by same-semantics arcs.

    Args:
        semantics: The semantics shared in the subgraph.
        nodes: The nodes belonging to the subgraph.
        edges: The edges belonging to the subgraph.
        entrying: The edges starting from the upstream frontier of the subgraph.
        exiting: The edges ending into the downstream frontier of the subgraph.
        dominators: The nodes that dominate the traffic through the subgraph.

    """

    semantics: _semantics.Semantics
    nodes: set[_graph.Node] = dataclasses.field(default_factory=set)
    edges: set[_graph.Edge] = dataclasses.field(default_factory=set)
    entrying: set[_graph.Edge] = dataclasses.field(default_factory=set)
    exiting: set[_graph.Edge] = dataclasses.field(default_factory=set)
    dominators: set[_graph.Node] = dataclasses.field(default_factory=set)

    @property
    def entries(self) -> set[_graph.Node]:
        """The subgraph's input frontier."""
        return {pred for (pred, _, (_, _)) in self.entrying}

    @property
    def exits(self) -> set[_graph.Node]:
        """The subgraph's output frontier."""
        return {succ for (_, succ, (_, _)) in self.exiting}

    @property
    def dominating(self) -> set[_graph.Edge]:
        """The edges ending into the subgraph's dominators."""
        return {(u, v, (p, q)) for (u, v, (p, q)) in self.edges if (v in self.dominators)}


_KEY_EDGE_SUBGRAPH: str = "subgraph"
"""The key under which the subgraph to which the edge itself belongs is stored."""


def has_subgraph(graph: _graph.Graph, edge: _graph.Edge) -> bool:
    """Verify whether an edge has a subgraph."""
    return _KEY_EDGE_SUBGRAPH in graph.edges[edge]


def set_subgraph(graph: _graph.Graph, edge: _graph.Edge, subgraph: Subgraph) -> None:
    """Write an edge's subgraph."""
    graph.edges[edge][_KEY_EDGE_SUBGRAPH] = subgraph


def get_subgraph(graph: _graph.Graph, edge: _graph.Edge) -> Subgraph:
    """Read an edge's subgraph."""
    return cast(Subgraph, graph.edges[edge][_KEY_EDGE_SUBGRAPH])
