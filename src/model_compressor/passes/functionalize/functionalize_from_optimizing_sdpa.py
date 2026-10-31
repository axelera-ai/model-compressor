# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import cast

import torch
import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base

import model_compressor.nn as nn
import model_compressor.ops  # noqa: F401 - register the xaten custom operators
import model_compressor.passes.annotate as annotate


class FunctionalizeFromOptimizingSDPA(pass_base.PassBase):
    """Convert each `OptimizingSDPA` modular operation into an equivalent true-quantized functional graph."""

    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        # find the matches
        matches = []
        for node in graph_module.graph.nodes:
            if (node.op == "call_module") and isinstance(graph_module.get_submodule(node.target), nn.OptimizingSDPA):
                matches.append((node,))
        # rewrite the matches
        for match in matches:
            (old_node,) = match
            old_module = cast(nn.OptimizingSDPA, graph_module.get_submodule(old_node.target))
            new_target = old_node.name.replace("_optimizing_module", "")
            # insert the functional operator nodes
            with graph_module.graph.inserting_before(old_node):
                new_node_query = graph_module.graph.call_function(
                    torch.ops.xaten.encode.default,
                    args=(),
                    kwargs={"input": old_node.kwargs["query"]},
                    name=f"{new_target}_query_encoder",
                )
            with graph_module.graph.inserting_before(old_node):
                new_node_key = graph_module.graph.call_function(
                    torch.ops.xaten.encode.default,
                    args=(),
                    kwargs={"input": old_node.kwargs["key"]},
                    name=f"{new_target}_key_encoder",
                )
            with graph_module.graph.inserting_before(old_node):
                new_node_value = graph_module.graph.call_function(
                    torch.ops.xaten.encode.default,
                    args=(),
                    kwargs={"input": old_node.kwargs["value"]},
                    name=f"{new_target}_value_encoder",
                )
            new_kwargs_sdpa = {
                "query": new_node_query,
                "key": new_node_key,
                "value": new_node_value,
                "attn_mask": old_node.kwargs["attention_mask"],
                "dropout_p": old_module.dropout_p,
                "is_causal": old_module.is_causal,
                "scale": old_module.scale,
                "enable_gqa": old_module.enable_gqa,
            }
            with graph_module.graph.inserting_before(old_node):
                new_node_sdpa = graph_module.graph.call_function(
                    torch.ops.aten.scaled_dot_product_attention.default,
                    args=(),
                    kwargs=new_kwargs_sdpa,
                    name=new_target,
                )
            # remove the modular operation node
            old_node.replace_all_uses_with(new_node_sdpa)
            if len(old_node.users) == 0:
                graph_module.graph.erase_node(old_node)
                graph_module.delete_submodule(old_node.target)
            # annotate the functional operator node
            old_annotation = annotate.get_annotation(old_node)
            new_annotation = annotate.Annotation(
                source=old_annotation.chosen,
                chosen=old_annotation.chosen,
            )
            annotate.set_annotation(new_node_sdpa, new_annotation)
        if len(matches) == 0:  # nothing to rewrite: exit early
            return pass_base.PassResult(graph_module, False)
        # consolidate the changes
        graph_module.graph.lint()
        graph_module.recompile()
        return pass_base.PassResult(graph_module, True)
