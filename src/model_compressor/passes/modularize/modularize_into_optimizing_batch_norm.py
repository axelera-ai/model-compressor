# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import cast

import torch
import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base

import model_compressor.nn as nn
import model_compressor.passes.annotate as annotate
import model_compressor.passes.utils.rgetattr as rgetattr


class ModularizeIntoOptimizingBatchNorm(pass_base.PassBase):
    """Convert each `aten._native_batch_norm_legit_no_training.default` functional operator into
    an `OptimizingBatchNorm` modular operation.

    The functional op is multi-return (`(Tensor, Tensor, Tensor)`), but only the first return is
    meaningful in eval mode (the latter two are zero-element placeholders carried over from the
    training-time op's ABI). The modular form preserves the 3-tuple return shape so that the
    canonical-form `getitem(BN, 0)` consumers continue to work unchanged.

    """

    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        target = torch.ops.aten._native_batch_norm_legit_no_training.default
        # find the matches
        matches = []
        for node in graph_module.graph.nodes:
            if (node.op == "call_function") and (node.target is target):
                kwargs = node.kwargs
                if all(
                    isinstance(kwargs[name], fx.Node) and (kwargs[name].op == "get_attr")
                    for name in ("weight", "bias", "running_mean", "running_var")
                ):
                    matches.append((node,))
        # rewrite the matches
        if len(matches) == 0:
            return pass_base.PassResult(graph_module, False)
        for match in matches:
            (old_node,) = match
            # create the modular operation
            weight_node = old_node.kwargs["weight"]
            bias_node = old_node.kwargs["bias"]
            running_mean_node = old_node.kwargs["running_mean"]
            running_var_node = old_node.kwargs["running_var"]
            weight_tensor = rgetattr.rgetattr(graph_module, weight_node)
            bias_tensor = rgetattr.rgetattr(graph_module, bias_node)
            running_mean_tensor = rgetattr.rgetattr(graph_module, running_mean_node)
            running_var_tensor = rgetattr.rgetattr(graph_module, running_var_node)
            (channels,) = weight_tensor.shape
            new_module = nn.OptimizingBatchNorm(
                channels=channels,
                momentum=cast(float, old_node.kwargs["momentum"]),
                eps=cast(float, old_node.kwargs["eps"]),
            )
            with torch.no_grad():
                _ = cast(torch.Tensor, new_module.weight).data.copy_(weight_tensor)
                _ = cast(torch.Tensor, new_module.bias).data.copy_(bias_tensor)
                _ = new_module.running_mean.data.copy_(running_mean_tensor)
                _ = new_module.running_var.data.copy_(running_var_tensor)
            # insert the modular operation node
            new_target = f"{old_node.name}_optimizing_module"
            graph_module.register_module(new_target, new_module)
            new_kwargs = {"input": old_node.kwargs["input"]}
            with graph_module.graph.inserting_before(old_node):
                new_node = graph_module.graph.call_module(new_target, args=(), kwargs=new_kwargs)
            # uninstall the functional operator node
            old_node.replace_all_uses_with(new_node)
            if len(old_node.users) == 0:
                graph_module.graph.erase_node(old_node)
            if len(weight_node.users) == 0:
                graph_module.graph.erase_node(weight_node)
            if len(bias_node.users) == 0:
                graph_module.graph.erase_node(bias_node)
            if len(running_mean_node.users) == 0:
                graph_module.graph.erase_node(running_mean_node)
            if len(running_var_node.users) == 0:
                graph_module.graph.erase_node(running_var_node)
            # annotate the modular operation node
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
