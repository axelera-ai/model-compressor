# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch
import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base

import model_compressor.nn as nn
import model_compressor.passes.annotate as annotate
import model_compressor.passes.utils.rgetattr as rgetattr


class ModularizeIntoOptimizingLinear(pass_base.PassBase):
    """Convert each `aten.linear.default` functional operator into an `OptimizingLinear` modular operation."""

    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        # find the matches
        matches = []
        for node in graph_module.graph.nodes:
            if (node.op == "call_function") and (node.target is torch.ops.aten.linear.default):
                weight_node, bias_node_or_none = node.kwargs["weight"], node.kwargs["bias"]
                if (weight_node.op == "get_attr") and (bias_node_or_none is None):
                    matches.append((node,))
        # rewrite the matches
        if len(matches) == 0:  # nothing to rewrite: exit early
            return pass_base.PassResult(graph_module, False)
        for match in matches:
            (old_node,) = match
            # create the modular operation
            weight_node = old_node.kwargs["weight"]
            weight_tensor = rgetattr.rgetattr(graph_module, weight_node)
            out_features, in_features = weight_tensor.shape
            new_module = nn.OptimizingLinear(in_features=in_features, out_features=out_features)
            with torch.no_grad():
                _ = new_module.weight.data.copy_(weight_tensor)
            # insert the modular operation node
            new_target = f"{old_node.name}_optimizing_module"
            graph_module.register_module(new_target, new_module)
            new_kwargs = {"input": old_node.kwargs["input"]}
            with graph_module.graph.inserting_before(old_node):
                new_node = graph_module.graph.call_module(new_target, args=(), kwargs=new_kwargs)
            # remove the functional operator node
            old_node.replace_all_uses_with(new_node)
            if len(old_node.users) == 0:
                graph_module.graph.erase_node(old_node)
            if len(weight_node.users) == 0:
                graph_module.graph.erase_node(weight_node)
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
