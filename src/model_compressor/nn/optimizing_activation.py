# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import Any, cast

import torch
import torch.nn as nn


class OptimizingActivation(nn.Module):
    def __init__(self, *, op: torch._ops.OpOverload, name_to_parameter: dict[str, Any]) -> None:
        super().__init__()
        self.op = op
        self.name_to_parameter = name_to_parameter

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.op(input, **self.name_to_parameter))
