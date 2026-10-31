# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn


class Observer(nn.Module):
    """Hook-installable operation to collect statistics."""

    def forward(self, module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        pass
