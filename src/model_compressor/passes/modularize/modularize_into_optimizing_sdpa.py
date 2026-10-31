# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch
import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base

import model_compressor.nn as nn
import model_compressor.ops  # noqa: F401 - register the xaten custom operators
import model_compressor.passes.annotate as annotate


class ModularizeIntoOptimizingSDPA(pass_base.PassBase):
    """Convert each `aten.scaled_dot_product_attention.default` functional operator into an `OptimizingSDPA` modular operation."""

    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        # find the matches
        matches = []
        for node in graph_module.graph.nodes:
            if (node.op == "call_function") and (node.target is torch.ops.aten.scaled_dot_product_attention.default):
                matches.append((node,))
        # rewrite the matches
        if len(matches) == 0:  # nothing to rewrite: exit early
            return pass_base.PassResult(graph_module, False)
        for match in matches:
            (old_node,) = match
            # create the modular operation
            new_module = nn.OptimizingSDPA(
                dropout_p=old_node.kwargs["dropout_p"],
                is_causal=old_node.kwargs["is_causal"],
                scale=old_node.kwargs["scale"],
                enable_gqa=old_node.kwargs["enable_gqa"],
            )
            # insert the modular operation node
            new_target = f"{old_node.name}_optimizing_module"
            graph_module.register_module(new_target, new_module)
            new_kwargs = {
                "query": old_node.kwargs["query"],
                "key": old_node.kwargs["key"],
                "value": old_node.kwargs["value"],
                "attention_mask": old_node.kwargs["attn_mask"],
            }
            with graph_module.graph.inserting_before(old_node):
                new_node = graph_module.graph.call_module(new_target, args=(), kwargs=new_kwargs)
            # remove the functional operator node
            old_node.replace_all_uses_with(new_node)
            if len(old_node.users) == 0:
                graph_module.graph.erase_node(old_node)
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
