# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import functools

import torch
import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base


class CanonicalizeBias(pass_base.PassBase):
    """Reshape convolution bias parameters to broadcast against the convolution output.

    A convolution bias is a rank-1 ``(C,)`` parameter, which does not broadcast against the
    convolution's ``(N, C, *spatial)`` output once the bias is materialized as a separate addition (as
    done by ``CanonicalizeConvolution``). This pass rewrites each such bias parameter, in place, to the
    broadcast shape ``(1, C, *1)`` - one trailing singleton per spatial dimension, derived from the
    weight rank (``out_channels, in_channels, *kernel``) rather than the padding, which is not a reliable
    indicator of spatial rank. The rewrite is compile-time only - it mutates the stored parameter rather
    than inserting a runtime reshape operator - and assumes that a bias parameter is consumed by exactly
    one convolution, so that mutating it in place is safe. It must run before ``CanonicalizeConvolution``.
    """

    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        def _is_get_attr_node(node: object) -> bool:
            return isinstance(node, fx.Node) and (node.op == "get_attr")

        def _get_parameter(target: str) -> torch.Tensor:
            *parents, leaf = target.split(".")
            return getattr(functools.reduce(getattr, parents, graph_module), leaf)

        def _set_parameter(target: str, value: torch.Tensor) -> None:
            *parents, leaf = target.split(".")
            setattr(functools.reduce(getattr, parents, graph_module), leaf, value)

        changed = False
        for node in graph_module.graph.nodes:
            if (node.op != "call_function") or (node.target is not torch.ops.aten.convolution.default):
                continue
            bias_node, weight_node = node.kwargs["bias"], node.kwargs["weight"]
            if not _is_get_attr_node(bias_node):
                continue
            # a bias parameter is consumed by exactly one convolution, so it is safe to mutate it in place
            assert len(bias_node.users) == 1
            number_of_spatial_dimensions = _get_parameter(weight_node.target).ndim - 2
            shape = (1, -1, *(1 for _ in range(number_of_spatial_dimensions)))
            bias = _get_parameter(bias_node.target)
            reshaped = bias.reshape(*shape)
            if isinstance(bias, torch.nn.Parameter):
                reshaped = torch.nn.Parameter(reshaped, requires_grad=bias.requires_grad)
            _set_parameter(bias_node.target, reshaped)
            if "val" in bias_node.meta:
                bias_node.meta["val"] = bias_node.meta["val"].reshape(*shape)
            changed = True
        return pass_base.PassResult(graph_module, changed)
