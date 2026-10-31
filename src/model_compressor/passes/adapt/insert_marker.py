# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base

import model_compressor.passes.adapt.base as base
import model_compressor.passes.adapt.insert_markers as insert_markers
import model_compressor.passes.utils.state as _state


class InsertMarker(pass_base.PassBase):
    """Insert codec marker operators around functional nodes."""

    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        """Insert codec marker operators around each functional node in an ``fx.GraphModule``.

        Args:
            graph_module: An ``fx.GraphModule``.

        Returns:
            A pair whose first item is the ``fx.GraphModule`` where codec marker operators have
                been inserted, and whose second item is a flag indicating whether any codec marker
                operator was inserted.

        """
        state = _state.State(graph_module.graph)
        graph = base.convert_fx_to_nx(graph_module)
        graph, modified = insert_markers.insert_markers(state, graph)
        graph_module = base.convert_nx_to_fx(graph)
        return pass_base.PassResult(graph_module, modified)
