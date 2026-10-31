# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from collections.abc import Hashable
from typing import NamedTuple, TypeAlias

import networkx as nx

Node: TypeAlias = Hashable
"""A node in a computational graph.

It represents a data source (input, parameter/buffer), a data sink (output), or an operator.

"""

Port: TypeAlias = tuple[str] | tuple[str, int]
"""The input or output port of an operator."""

Edge: TypeAlias = tuple[Node, Node, tuple[Port, Port]]
"""An edge in a computational graph.

It represents a read dependency, from an upstream port to a downstream port.

"""

Graph: TypeAlias = nx.MultiDiGraph
"""A computational graph."""


class PassResult(NamedTuple):
    """A pair containing a graph and a flag indicating whether it was modified."""

    graph: Graph
    modified: bool
