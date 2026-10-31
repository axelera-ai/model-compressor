# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import cast

import torch
import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base

import model_compressor.nn as nn
import model_compressor.passes.annotate as annotate


class FunctionalizeFromOptimizingBatchNorm(pass_base.PassBase):
    """Convert each `OptimizingBatchNorm` modular operation back into the equivalent functional
    `aten._native_batch_norm_legit_no_training.default` call.

    Batch normalization stays in floating-point in this version - no quantization wrap is
    applied around it. The functional form is the same multi-return op the source graph carried;
    the canonical-form `getitem(BN, 0)` consumers continue to work unchanged.

    """

    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        # find the matches
        matches = []
        for node in graph_module.graph.nodes:
            if (node.op == "call_module") and isinstance(
                graph_module.get_submodule(node.target), nn.OptimizingBatchNorm
            ):
                matches.append((node,))
        # rewrite the matches
        if len(matches) == 0:
            return pass_base.PassResult(graph_module, False)
        for match in matches:
            (old_node,) = match
            # insert the buffer/parameter nodes
            new_target = old_node.name.replace("_optimizing_module", "")
            old_module = cast(nn.OptimizingBatchNorm, graph_module.get_submodule(old_node.target))
            graph_module.register_parameter(
                f"{new_target}_weight",
                torch.nn.Parameter(cast(torch.Tensor, old_module.weight).data.detach().clone()),
            )
            graph_module.register_parameter(
                f"{new_target}_bias",
                torch.nn.Parameter(cast(torch.Tensor, old_module.bias).data.detach().clone()),
            )
            graph_module.register_buffer(f"{new_target}_running_mean", old_module.running_mean.detach().clone())
            graph_module.register_buffer(f"{new_target}_running_var", old_module.running_var.detach().clone())
            with graph_module.graph.inserting_before(old_node):
                node_weight = graph_module.graph.get_attr(f"{new_target}_weight")
                node_bias = graph_module.graph.get_attr(f"{new_target}_bias")
                node_running_mean = graph_module.graph.get_attr(f"{new_target}_running_mean")
                node_running_var = graph_module.graph.get_attr(f"{new_target}_running_var")
            # insert the functional operator node
            with graph_module.graph.inserting_before(old_node):
                new_node = graph_module.graph.call_function(
                    torch.ops.aten._native_batch_norm_legit_no_training.default,
                    args=(),
                    kwargs={
                        "input": old_node.kwargs["input"],
                        "weight": node_weight,
                        "bias": node_bias,
                        "running_mean": node_running_mean,
                        "running_var": node_running_var,
                        "momentum": old_module.momentum,
                        "eps": old_module.eps,
                    },
                    name=new_target,
                )
            # remove the modular operation node
            old_node.replace_all_uses_with(new_node)
            if len(old_node.users) == 0:
                graph_module.graph.erase_node(old_node)
                graph_module.delete_submodule(old_node.target)
            # annotate the functional operator node
            old_annotation = annotate.get_annotation(old_node)
            new_annotation = annotate.Annotation(
                source=old_annotation.chosen,
                chosen=old_annotation.chosen,
            )
            annotate.set_annotation(new_node, new_annotation)
        # consolidate the changes
        graph_module.graph.lint()
        graph_module.recompile()
        return pass_base.PassResult(graph_module, True)
