# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch
import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base

import model_compressor.ops  # noqa: F401 - register extended ATen (XATen) operators


def _has_single_non_container_output(target: torch._ops.OpOverload) -> bool:
    if len(target._schema.returns) != 1:
        return False
    output, *_ = target._schema.returns
    return type(output.type) in (torch.IntType, torch.SymIntType, torch.TensorType)


class ReifyFanOut(pass_base.PassBase):
    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        # find the matches
        matches = []
        for node in graph_module.graph.nodes:
            if (
                (node.op == "call_function")
                and isinstance(node.target, torch._ops.OpOverload)
                and _has_single_non_container_output(node.target)
                and (len(node.users) > 1)
            ):
                matches.append((node,))
        # rewrite the matches
        if len(matches) == 0:  # nothing to rewrite: exit early
            return pass_base.PassResult(graph_module, False)
        for match in matches:
            (node,) = match
            output, *_ = node.target._schema.returns
            match type(output.type):
                case torch.IntType | torch.SymIntType:
                    target = torch.ops.xaten.fan_out.int
                case torch.TensorType:
                    target = torch.ops.xaten.fan_out.Tensor
                case _:
                    raise ValueError
            with graph_module.graph.inserting_after(node):
                # keep `kwargs` empty, so that rewiring consumers does not create self-loops
                new_node_fan_out = graph_module.graph.call_function(target, args=(), kwargs={})
            node.replace_all_uses_with(new_node_fan_out)
            new_node_fan_out.kwargs = torch.fx.immutable_collections.immutable_dict({"input": node})
        # consolidate the changes
        graph_module.graph.lint()
        graph_module.recompile()
        return pass_base.PassResult(graph_module, True)
