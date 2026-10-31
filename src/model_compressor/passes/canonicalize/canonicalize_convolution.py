# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from collections.abc import Sequence

import torch
import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base


class CanonicalizeConvolution(pass_base.PassBase):
    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        # find the matches
        def _is_get_attr_node(node: fx.Node) -> bool:
            return isinstance(node, fx.Node) and (node.op == "get_attr")

        def _is_non_null_padding(padding: Sequence[int]) -> bool:
            return any((item != 0) for item in padding)

        matches = []
        for node in graph_module.graph.nodes:
            if (node.op == "call_function") and (node.target is torch.ops.aten.convolution.default):
                weight_node, bias_node_or_none, padding = (
                    node.kwargs["weight"],
                    node.kwargs["bias"],
                    node.kwargs["padding"],
                )
                if _is_get_attr_node(weight_node) and (
                    _is_get_attr_node(bias_node_or_none) or _is_non_null_padding(padding)
                ):
                    matches.append((node,))
        # rewrite the matches
        if len(matches) == 0:  # nothing to rewrite: exit early
            return pass_base.PassResult(graph_module, False)
        for match in matches:
            (old_node,) = match
            (
                input_node,
                weight_node,
                bias_node_or_none,
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
            ) = (
                old_node.kwargs["input"],
                old_node.kwargs["weight"],
                old_node.kwargs["bias"],
                old_node.kwargs["stride"],
                old_node.kwargs["padding"],
                old_node.kwargs["dilation"],
                old_node.kwargs["transposed"],
                old_node.kwargs["output_padding"],
                old_node.kwargs["groups"],
            )
            has_bias = _is_get_attr_node(bias_node_or_none)
            has_padding = _is_non_null_padding(padding)
            match (has_bias, has_padding):
                case (True, True):
                    # insert the canonical functional operator nodes
                    pad = [v for p in reversed(padding) for v in (p, p)]
                    with graph_module.graph.inserting_before(old_node):
                        new_node_constant_pad_nd = graph_module.graph.call_function(
                            torch.ops.aten.constant_pad_nd.default,
                            args=(),
                            kwargs={"self": input_node, "pad": pad, "value": 0.0},
                            name=f"{old_node.name}_canonical_constant_pad_nd",
                        )
                    with graph_module.graph.inserting_before(old_node):
                        new_node_convolution = graph_module.graph.call_function(
                            torch.ops.aten.convolution.default,
                            args=(),
                            kwargs={
                                "input": new_node_constant_pad_nd,
                                "weight": weight_node,
                                "bias": None,
                                "stride": stride,
                                "padding": [0 for _ in padding],
                                "dilation": dilation,
                                "transposed": transposed,
                                "output_padding": output_padding,
                                "groups": groups,
                            },
                            name=f"{old_node.name}_canonical_convolution",
                        )
                    with graph_module.graph.inserting_before(old_node):
                        new_node_add = graph_module.graph.call_function(
                            torch.ops.aten.add.Tensor,
                            args=(),
                            kwargs={
                                "self": new_node_convolution,
                                "other": bias_node_or_none,
                                "alpha": 1.0,
                            },
                            name=f"{old_node.name}_canonical_add",
                        )
                    # remove the non-canonical functional operator node
                    old_node.replace_all_uses_with(new_node_add)
                    if len(old_node.users) == 0:
                        graph_module.graph.erase_node(old_node)
                case (True, False):
                    # insert the canonical functional operator nodes
                    with graph_module.graph.inserting_before(old_node):
                        new_node_convolution = graph_module.graph.call_function(
                            torch.ops.aten.convolution.default,
                            args=(),
                            kwargs={
                                "input": input_node,
                                "weight": weight_node,
                                "bias": None,
                                "stride": stride,
                                "padding": padding,
                                "dilation": dilation,
                                "transposed": transposed,
                                "output_padding": output_padding,
                                "groups": groups,
                            },
                            name=f"{old_node.name}_canonical_convolution",
                        )
                    with graph_module.graph.inserting_before(old_node):
                        new_node_add = graph_module.graph.call_function(
                            torch.ops.aten.add.Tensor,
                            args=(),
                            kwargs={
                                "self": new_node_convolution,
                                "other": bias_node_or_none,
                                "alpha": 1.0,
                            },
                            name=f"{old_node.name}_canonical_add",
                        )
                    # remove the non-canonical functional operator node
                    old_node.replace_all_uses_with(new_node_add)
                    if len(old_node.users) == 0:
                        graph_module.graph.erase_node(old_node)
                case (False, True):
                    # insert the canonical functional operator nodes
                    pad = [v for p in reversed(padding) for v in (p, p)]
                    with graph_module.graph.inserting_before(old_node):
                        new_node_constant_pad_nd = graph_module.graph.call_function(
                            torch.ops.aten.constant_pad_nd.default,
                            args=(),
                            kwargs={"self": input_node, "pad": pad, "value": 0.0},
                            name=f"{old_node.name}_canonical_constant_pad_nd",
                        )
                    with graph_module.graph.inserting_before(old_node):
                        new_node_convolution = graph_module.graph.call_function(
                            torch.ops.aten.convolution.default,
                            args=(),
                            kwargs={
                                "input": new_node_constant_pad_nd,
                                "weight": weight_node,
                                "bias": bias_node_or_none,
                                "stride": stride,
                                "padding": [0 for _ in padding],
                                "dilation": dilation,
                                "transposed": transposed,
                                "output_padding": output_padding,
                                "groups": groups,
                            },
                            name=f"{old_node.name}_canonical_convolution",
                        )
                    # remove the non-canonical functional operator node
                    old_node.replace_all_uses_with(new_node_convolution)
                    if len(old_node.users) == 0:
                        graph_module.graph.erase_node(old_node)
                case _:
                    raise RuntimeError(
                        f"Node {node!r} was matched for canonicalization, but has zero padding and no bias"
                    )
        # consolidate the changes
        graph_module.graph.lint()
        graph_module.recompile()
        return pass_base.PassResult(graph_module, True)
