# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import dataclasses
from typing import TypeAlias, cast

import torch
import torch.fx as fx

import model_compressor.passes.adapt.base.fx.semantics as _semantics


@dataclasses.dataclass
class Subgraph:
    """A subgraph sharing same-semantics data.

    !!! note

        An ``fx.Graph`` does not represent edges explicitly, so subgraph edges are represented as
        dictionaries mapping ``fx.Node``s to their keyword arguments that represent input data
        with the subgraph's semantics.

    Args:
        semantics: The semantics shared in the subgraph.
        nodes: The nodes belonging to the subgraph.
        edges: The edges belonging to the subgraph.
        entries: The subgraph's input frontier.
        exits: The subgraph's output frontier.
        dominators: The nodes that dominate the traffic through the subgraph.
        entrying: The edges starting from the upstream frontier of the subgraph.
        exiting: The edges ending into the downstream frontier of the subgraph.
        dominating: The edges ending into the subgraph's dominators.

    """

    semantics: _semantics.Semantics
    nodes: set[fx.Node]
    edges: dict[fx.Node, torch.fx.immutable_collections.immutable_dict]
    entries: set[fx.Node]
    exits: set[fx.Node]
    dominators: set[fx.Node]
    entrying: dict[fx.Node, torch.fx.immutable_collections.immutable_dict]
    exiting: dict[fx.Node, torch.fx.immutable_collections.immutable_dict]
    dominating: dict[fx.Node, torch.fx.immutable_collections.immutable_dict]


KWArgToSubgraph: TypeAlias = dict[str, Subgraph]
"""A mapping from keyword arguments to same-semantics subgraphs."""


_KEY_META_KWARG_TO_SUBGRAPH: str = "kwarg-to-subgraph"
"""The key under which a node's keyword-to-subgraph mapping is stored."""


def has_kwarg_to_subgraph(node: fx.Node) -> bool:
    """Verify whether a node has a keyword-to-subgraph mapping."""
    return _KEY_META_KWARG_TO_SUBGRAPH in node.meta


def set_kwarg_to_subgraph(node: fx.Node, kwarg_to_subgraph: KWArgToSubgraph) -> None:
    """Write a node's keyword-to-subgraph mapping."""
    node.meta[_KEY_META_KWARG_TO_SUBGRAPH] = kwarg_to_subgraph


def get_kwarg_to_subgraph(node: fx.Node) -> KWArgToSubgraph:
    """Read a node's keyword-to-subgraph mapping."""
    return cast(KWArgToSubgraph, node.meta[_KEY_META_KWARG_TO_SUBGRAPH])
