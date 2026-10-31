# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch.fx as fx
import torch.fx.passes.infra.pass_manager as _pass_manager

import model_compressor.api.config as _config
import model_compressor.passes as _passes


def prepare_before_optimization(graph_module: fx.GraphModule, config: _config.Config) -> fx.GraphModule:
    passes = []
    # register the canonicalization passes
    passes.extend(
        [
            _passes.canonicalize.ReifyFanOut(),
            _passes.canonicalize.CanonicalizeLinear(),
            _passes.canonicalize.CanonicalizeBias(),
            _passes.canonicalize.CanonicalizeConvolution(),
            _passes.canonicalize.CanonicalizeMaxPool2dWithIndices(),
        ]
    )
    # register the annotation passes
    passes.extend(
        [
            _passes.annotate.Annotate(
                input=config.annotation.input, database=config.annotation.database, picker=config.annotation.picker
            ),
        ]
    )
    # register the adaptation passes
    passes.extend(
        [
            _passes.adapt.InsertMarker(),
            _passes.adapt.ComputeKWArgToSubgraph(config.adaptation.assignment, config.adaptation.coalescence),
            _passes.adapt.Materialize(),
        ]
    )
    # # register the modularization passes
    # passes.extend([rule.modularize() for rule in config.optimization.rules])
    # transform
    manager = _pass_manager.PassManager(passes)
    graph_module = manager(graph_module).graph_module
    return graph_module
