# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch
import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base

import model_compressor.nn as nn
import model_compressor.passes.annotate as annotate
import model_compressor.passes.utils.rgetattr as rgetattr


class ModularizeIntoOptimizingConvolution(pass_base.PassBase):
    """Convert each canonical `aten.convolution.default` functional operator into an
    `OptimizingConvolution` modular operation.

    Assumes `canonicalize_convolution` has already run, so the matched node has ``bias is None``
    and ``padding`` entirely zero (any non-zero padding has been extracted upstream into a
    preceding `constant_pad_nd` node, and any non-`None` bias has been extracted downstream
    into a following `aten.add.Tensor` node).

    """

    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        # find the matches
        matches = []
        for node in graph_module.graph.nodes:
            if (node.op == "call_function") and (node.target is torch.ops.aten.convolution.default):
                weight_node = node.kwargs["weight"]
                bias_node_or_none = node.kwargs["bias"]
                padding = node.kwargs["padding"]
                if (
                    isinstance(weight_node, fx.Node)
                    and (weight_node.op == "get_attr")
                    and (bias_node_or_none is None)
                    and all((p == 0) for p in padding)
                ):
                    matches.append((node,))
        # rewrite the matches
        if len(matches) == 0:
            return pass_base.PassResult(graph_module, False)
        for match in matches:
            (old_node,) = match
            # create the modular operation
            weight_node = old_node.kwargs["weight"]
            weight_tensor = rgetattr.rgetattr(graph_module, weight_node)
            out_channels, in_per_group, *kernel_size = weight_tensor.shape
            groups = old_node.kwargs["groups"]
            in_channels = in_per_group * groups
            new_module = nn.OptimizingConvolution(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=list(kernel_size),
                stride=list(old_node.kwargs["stride"]),
                dilation=list(old_node.kwargs["dilation"]),
                transposed=old_node.kwargs["transposed"],
                output_padding=list(old_node.kwargs["output_padding"]),
                groups=groups,
            )
            with torch.no_grad():
                _ = new_module.weight.data.copy_(weight_tensor)
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
