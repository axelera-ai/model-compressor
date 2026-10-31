# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import cast

import torch
import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base

import model_compressor.nn as nn
import model_compressor.ops  # noqa: F401 - register the xaten custom operators
import model_compressor.passes.annotate as annotate


class FunctionalizeFromOptimizingConvolution(pass_base.PassBase):
    """Convert each `OptimizingConvolution` modular operation into an equivalent true-quantized
    functional graph: ``encode(act) -> encode(weight) -> aten.convolution -> decode``.

    Mirrors the structure of `FunctionalizeFromOptimizingLinear` exactly; the only difference is
    the spatial conv arguments (`stride`, `padding`, `dilation`, `transposed`, `output_padding`,
    `groups`) carried over from the `OptimizingConvolution` module instance.

    """

    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        # find the matches
        matches = []
        for node in graph_module.graph.nodes:
            if (node.op == "call_module") and isinstance(
                graph_module.get_submodule(node.target), nn.OptimizingConvolution
            ):
                matches.append((node,))
        # rewrite the matches
        if len(matches) == 0:
            return pass_base.PassResult(graph_module, False)
        for match in matches:
            (old_node,) = match
            # insert the weight parameter node
            new_target = old_node.name.replace("_optimizing_module", "")
            old_module = cast(nn.OptimizingConvolution, graph_module.get_submodule(old_node.target))
            graph_module.register_buffer(f"{new_target}_weight", old_module.weight.data.detach().clone())
            with graph_module.graph.inserting_before(old_node):
                new_node_weight = graph_module.graph.get_attr(f"{new_target}_weight")
            # insert the functional operator nodes
            with graph_module.graph.inserting_before(old_node):
                new_node_input = graph_module.graph.call_function(
                    torch.ops.xaten.encode.default,
                    args=(),
                    kwargs={"input": old_node.kwargs["input"]},
                    name=f"{new_target}_input_encoder",
                )
            with graph_module.graph.inserting_before(old_node):
                new_node_weight = graph_module.graph.call_function(
                    torch.ops.xaten.encode.default,
                    args=(),
                    kwargs={"input": new_node_weight},
                    name=f"{new_target}_weight_encoder",
                )
            with graph_module.graph.inserting_before(old_node):
                new_node_convolution = graph_module.graph.call_function(
                    torch.ops.aten.convolution.default,
                    args=(),
                    kwargs={
                        "input": new_node_input,
                        "weight": new_node_weight,
                        "bias": None,
                        "stride": list(old_module.stride),
                        "padding": list(old_module.padding),
                        "dilation": list(old_module.dilation),
                        "transposed": bool(old_module.transposed),
                        "output_padding": list(old_module.output_padding),
                        "groups": int(old_module.groups),
                    },
                    name=new_target,
                )
            with graph_module.graph.inserting_before(old_node):
                new_node_fused = graph_module.graph.call_function(
                    torch.ops.xaten.decode.default,
                    args=(),
                    kwargs={"input": new_node_convolution},
                    name=f"{new_target}_fused_decoder",
                )
            # remove the modular operation node
            old_node.replace_all_uses_with(new_node_fused)
            if len(old_node.users) == 0:
                graph_module.graph.erase_node(old_node)
                graph_module.delete_submodule(old_node.target)
            # annotate the functional operator node
            old_annotation = annotate.get_annotation(old_node)
            new_annotation = annotate.Annotation(
                source=old_annotation.chosen,
                chosen=old_annotation.chosen,
            )
            annotate.set_annotation(new_node_fused, new_annotation)
        # consolidate the changes
        graph_module.graph.lint()
        graph_module.recompile()
        return pass_base.PassResult(graph_module, True)
