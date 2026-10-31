# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import cast

import torch
import torch.nn as nn


class OptimizingRMSNorm(nn.Module):
    weight: nn.Parameter | None

    def __init__(self, normalized_shape: list[int], use_weight: bool | None, eps: float | None) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.register_parameter(
            "weight",
            None if ((use_weight is None) or (use_weight is False)) else nn.Parameter(torch.ones(normalized_shape)),
        )
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            torch.ops.aten.rms_norm.default(input, self.normalized_shape, self.weight, self.eps),
        )
