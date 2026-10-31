# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

"""Define schema pickers - strategies for selecting one candidate schema per node."""

from collections.abc import Callable
from typing import TypeAlias

import torch.fx as fx

import model_compressor.passes.annotate.base as base

Picker: TypeAlias = Callable[[fx.Node, list[base.SchemaProtocol]], base.SchemaProtocol]
"""A strategy for selecting a single schema from list of candidates."""


class PriorityPicker:
    def __init__(self, priority: list[base.SchemaProtocol]) -> None:
        """Initialize the picker.

        Args:
            priority: An ordering of schemas - most-preferred first.

        """
        self._priority = priority

    def __call__(self, node: fx.Node, candidates: list[base.SchemaProtocol]) -> base.SchemaProtocol:
        """Select the candidate schema with highest priority."""

        def _rank(schema: base.SchemaProtocol) -> int:
            try:
                return self._priority.index(schema)
            except ValueError:
                # Unranked candidates are assigned the same (lowest) priority; ties are solved by
                # the order in which they appear as candidates.
                return len(self._priority)

        return min(candidates, key=_rank)
