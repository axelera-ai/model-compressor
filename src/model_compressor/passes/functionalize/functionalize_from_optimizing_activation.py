# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch
import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base

import model_compressor.nn as nn
import model_compressor.ops  # noqa: F401 - register the xaten custom operators
import model_compressor.passes.annotate as annotate


class FunctionalizeFromOptimizingActivation(pass_base.PassBase):
    """Convert each `OptimizingActivation` modular operation into the corresponding target functional operator."""

    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        # find the matches
        matches = []
        for node in graph_module.graph.nodes:
            if (node.op == "call_module") and isinstance(
                graph_module.get_submodule(node.target), nn.OptimizingActivation
            ):
                matches.append((node,))
        # rewrite the matches
        if len(matches) == 0:  # nothing to rewrite: exit early
            return pass_base.PassResult(graph_module, False)
        for match in matches:
            (old_node,) = match
            new_target = old_node.name.replace("_optimizing_module", "")
            # insert the functional operator node
            with graph_module.graph.inserting_before(old_node):
                new_node = graph_module.graph.call_function(
                    torch.ops.xaten.approximate.default,
                    args=(),
                    kwargs={"input": old_node.kwargs["input"]},
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
