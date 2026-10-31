# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import cast

import torch
import torch.nn as nn


class OptimizingLinear(nn.Module):
    weight: nn.Parameter

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter("weight", nn.Parameter(torch.randn((out_features, in_features))))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, torch.ops.aten.linear.default(input, self.weight, None))
