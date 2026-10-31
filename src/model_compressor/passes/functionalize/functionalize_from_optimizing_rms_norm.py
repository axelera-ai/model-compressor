# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import cast

import torch
import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base

import model_compressor.nn as nn
import model_compressor.passes.annotate as annotate


class FunctionalizeFromOptimizingRMSNorm(pass_base.PassBase):
    """Convert each `OptimizingRMSNorm` modular operation into an equivalent true-quantized functional graph."""

    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        # find the matches
        matches = []
        for node in graph_module.graph.nodes:
            if (node.op == "call_module") and isinstance(graph_module.get_submodule(node.target), nn.OptimizingRMSNorm):
                matches.append((node,))
        # rewrite the matches
        if len(matches) == 0:  # nothing to rewrite: exit early
            return pass_base.PassResult(graph_module, False)
        for match in matches:
            (old_node,) = match
            # install the weight parameter node
            old_module = cast(nn.OptimizingRMSNorm, graph_module.get_submodule(old_node.target))
            new_target = old_node.name.replace("_optimizing_module", "")
            with torch.no_grad():
                graph_module.register_buffer(f"{new_target}_weight", cast(torch.Tensor, old_module.weight).data.clone())
            with graph_module.graph.inserting_before(old_node):
                new_node_weight = graph_module.graph.get_attr(f"{new_target}_weight")
            # insert the functional operator node
            new_kwargs = {
                "input": old_node.kwargs["input"],
                "normalized_shape": old_module.normalized_shape,
                "weight": new_node_weight,
                "eps": old_module.eps,
            }
            with graph_module.graph.inserting_before(old_node):
                new_node = graph_module.graph.call_function(
                    torch.ops.aten.rms_norm.default, args=(), kwargs=new_kwargs, name=new_target
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
