# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch.fx as fx
import torch.fx.passes.infra.pass_base as _pass_base
import torch.fx.passes.infra.pass_manager as _pass_manager

import model_compressor.api.config as _config


def finalize_after_optimization(graph_module: fx.GraphModule, config: _config.Config) -> fx.GraphModule:
    passes: list[_pass_base.PassBase] = []
    # # register the functionalization passes
    # passes.extend([rule.functionalize() for rule in config.optimization.rules])
    # transform
    pass_manager = _pass_manager.PassManager(passes)
    graph_module = pass_manager(graph_module).graph_module
    return graph_module
