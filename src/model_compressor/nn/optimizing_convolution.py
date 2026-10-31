# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import cast

import torch
import torch.nn as nn


class OptimizingConvolution(nn.Module):
    weight: nn.Parameter

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        transposed: bool,
        output_padding: list[int],
        groups: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = [0 for _ in kernel_size]
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.register_parameter(
            "weight", nn.Parameter(torch.randn((out_channels, in_channels // groups, *kernel_size)))
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            torch.ops.aten.convolution.default(
                input,
                self.weight,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.transposed,
                self.output_padding,
                self.groups,
            ),
        )
