# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import cast

import torch
import torch.nn as nn


class OptimizingBatchNorm(nn.Module):
    weight: nn.Parameter | None
    bias: nn.Parameter | None
    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(
        self,
        channels: int,
        momentum: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones((channels,))) if channels else None)
        self.register_parameter("bias", nn.Parameter(torch.zeros((channels,))) if channels else None)
        self.register_buffer("running_mean", torch.zeros((channels,)))
        self.register_buffer("running_var", torch.zeros((channels,)))
        self.momentum = momentum
        self.eps = eps

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            torch.ops.aten._native_batch_norm_legit_no_training.default(
                input,
                self.weight,
                self.bias,
                self.running_mean,
                self.running_var,
                self.momentum,
                self.eps,
            ),
        )
