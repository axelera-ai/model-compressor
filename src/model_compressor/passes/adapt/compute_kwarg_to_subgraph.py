# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import enum

import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base

import model_compressor.passes.adapt.assign_semantics as assign_semantics
import model_compressor.passes.adapt.base as base
import model_compressor.passes.adapt.coalesce_subgraphs as coalesce_subgraphs
import model_compressor.passes.adapt.induce_subgraphs as induce_subgraphs


class AssignmentMode(enum.Enum):
    """Configure the semantics assignment step."""

    STRICT = enum.auto()
    """Codec markers can commute only when they are lossless."""
    PERMISSIVE = enum.auto()
    """Codec markers can commute also when their I/O semantics are compatible."""


class CoalescenceMode(enum.Enum):
    """Configure the subgraph coalescence step."""

    WITHOUT = enum.auto()
    """Do not try to maximize the size of same-semantics subgraphs."""
    WITH = enum.auto()
    """Try to maximize the size of same-semantics subgraphs."""


class ComputeKWArgToSubgraph(pass_base.PassBase):
    """Assign semantics and compute same-semantics subgraphs."""

    def __init__(self, assignment_mode: AssignmentMode, coalescence_mode: CoalescenceMode) -> None:
        """Initialize the pass.

        Args:
            assignment_mode: Configure the semantics assignment (edge coloring) step.
            coalescence_mode: Configure the subgraph coalescence step.

        """
        self._assignment_mode = assignment_mode
        self._coalescence_mode = coalescence_mode

    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        """Assign semantics and compute same-semantics subgraphs in an ``fx.GraphModule``.

        Args:
            graph_module: An ``fx.GraphModule``.

        Returns:
            A pair whose first item is the ``fx.GraphModule`` and whose second item is a flag
                indicating whether semantics was assigned and same-semantics subgraphs were
                computed - the flag is unconditionally ``True``, since the assignment and subgraphs
                are recomputed unconditionally.

        """
        graph = base.convert_fx_to_nx(graph_module)
        match self._assignment_mode, self._coalescence_mode:
            case (AssignmentMode.STRICT, CoalescenceMode.WITHOUT):
                graph = assign_semantics.assign_semantics_strict(graph)
                graph = induce_subgraphs.induce_subgraphs(graph)
            case (AssignmentMode.STRICT, CoalescenceMode.WITH):
                graph = assign_semantics.assign_semantics_strict(graph)
                graph = induce_subgraphs.induce_subgraphs(graph)
                graph = coalesce_subgraphs.coalesce_subgraphs(graph)
            case (AssignmentMode.PERMISSIVE, CoalescenceMode.WITHOUT):
                graph = assign_semantics.assign_semantics_permissive(graph)
                graph = induce_subgraphs.induce_subgraphs(graph)
            case (AssignmentMode.PERMISSIVE, CoalescenceMode.WITH):
                graph = assign_semantics.assign_semantics_permissive(graph)
                graph = induce_subgraphs.induce_subgraphs(graph)
                graph = coalesce_subgraphs.coalesce_subgraphs(graph)
            case _:
                raise ValueError(f"Unknown modes {(self._assignment_mode, self._coalescence_mode)}")
        graph_module = base.convert_nx_to_fx(graph)
        graph_module = base.convert_nx_subgraphs_to_fx_subgraphs(graph, graph_module)
        return pass_base.PassResult(graph_module, True)
