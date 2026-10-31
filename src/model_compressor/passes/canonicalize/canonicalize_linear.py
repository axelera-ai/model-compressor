# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch
import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base


class CanonicalizeLinear(pass_base.PassBase):
    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        # find the matches
        def _is_get_attr_node(node: fx.Node) -> bool:
            return isinstance(node, fx.Node) and (node.op == "get_attr")

        matches = []
        for node in graph_module.graph.nodes:
            if (node.op == "call_function") and (node.target is torch.ops.aten.linear.default):
                weight_node, bias_node_or_none = node.kwargs["weight"], node.kwargs["bias"]
                if _is_get_attr_node(weight_node) and _is_get_attr_node(bias_node_or_none):
                    matches.append((node,))
        # rewrite the matches
        if len(matches) == 0:  # nothing to rewrite: exit early
            return pass_base.PassResult(graph_module, False)
        for match in matches:
            (old_node,) = match
            # insert the canonical functional operator nodes
            input_node, weight_node, bias_node = (
                old_node.kwargs["input"],
                old_node.kwargs["weight"],
                old_node.kwargs["bias"],
            )
            with graph_module.graph.inserting_before(old_node):
                new_node_linear = graph_module.graph.call_function(
                    torch.ops.aten.linear.default,
                    args=(),
                    kwargs={"input": input_node, "weight": weight_node, "bias": None},
                    name=f"{old_node.name}_canonical_linear",
                )
            with graph_module.graph.inserting_before(old_node):
                new_node_add = graph_module.graph.call_function(
                    torch.ops.aten.add.Tensor,
                    args=(),
                    kwargs={"self": new_node_linear, "other": bias_node, "alpha": 1.0},
                    name=f"{old_node.name}_canonical_add",
                )
            # remove the non-canonical functional operator node
            old_node.replace_all_uses_with(new_node_add)
            if len(old_node.users) == 0:
                graph_module.graph.erase_node(old_node)
        # consolidate the changes
        graph_module.graph.lint()
        graph_module.recompile()
        return pass_base.PassResult(graph_module, True)
