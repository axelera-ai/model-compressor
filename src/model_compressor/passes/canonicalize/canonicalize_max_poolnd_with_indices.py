# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch
import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base


class _CanonicalizeMaxPoolNdWithIndices(pass_base.PassBase):
    def __init__(self, target: torch._ops.OpOverload) -> None:
        self._target = target

    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        # find the matches
        matches = []
        for node in graph_module.graph.nodes:
            if (node.op == "call_function") and (node.target is self._target):
                padding = node.kwargs["padding"]
                if any((item != 0) for item in padding):
                    matches.append((node,))
        # rewrite the matches
        if len(matches) == 0:  # nothing to rewrite: exit early
            return pass_base.PassResult(graph_module, False)
        for match in matches:
            (old_node,) = match
            # insert the canonical functional operator nodes
            (
                input_node,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            ) = (
                old_node.kwargs["self"],
                old_node.kwargs["kernel_size"],
                old_node.kwargs["stride"],
                old_node.kwargs["padding"],
                old_node.kwargs["dilation"],
                old_node.kwargs["ceil_mode"],
            )
            # `max_pool` `padding` gives one symmetric value per spatial dimension, whereas
            # `constant_pad_nd` `pad` expects (prepend, append) pairs from the last dimension
            # backward - e.g. `[pad_h, pad_w]` becomes `[pad_w, pad_w, pad_h, pad_h]`.
            pad = [v for p in reversed(padding) for v in (p, p)]
            with graph_module.graph.inserting_before(old_node):
                new_node_constant_pad_nd = graph_module.graph.call_function(
                    torch.ops.aten.constant_pad_nd.default,
                    args=(),
                    kwargs={"self": input_node, "pad": pad, "value": 0.0},
                    name=f"{old_node.name}_canonical_constant_pad_nd",
                )
            with graph_module.graph.inserting_before(old_node):
                new_node_max_poolnd_with_indices = graph_module.graph.call_function(
                    self._target,
                    args=(),
                    kwargs={
                        "self": new_node_constant_pad_nd,
                        "kernel_size": kernel_size,
                        "stride": stride,
                        "padding": [0 for _ in padding],
                        "dilation": dilation,
                        "ceil_mode": ceil_mode,
                    },
                    name=f"{old_node.name}_canonical_{self._target.name().replace('aten::', '')}",
                )
            # remove the non-canonical functional operator node
            old_node.replace_all_uses_with(new_node_max_poolnd_with_indices)
            if len(old_node.users) == 0:
                graph_module.graph.erase_node(old_node)
        # consolidate the changes
        graph_module.graph.lint()
        graph_module.recompile()
        return pass_base.PassResult(graph_module, True)


class CanonicalizeMaxPool2dWithIndices(_CanonicalizeMaxPoolNdWithIndices):
    def __init__(self) -> None:
        super().__init__(target=torch.ops.aten.max_pool2d_with_indices.default)


class CanonicalizeMaxPool3dWithIndices(_CanonicalizeMaxPoolNdWithIndices):
    def __init__(self) -> None:
        super().__init__(target=torch.ops.aten.max_pool3d_with_indices.default)
