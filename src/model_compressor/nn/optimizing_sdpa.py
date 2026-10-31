# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import cast

import torch
import torch.nn as nn


class OptimizingSDPA(nn.Module):

    def __init__(
        self,
        dropout_p: float,
        is_causal: bool,
        *,
        scale: float | None,
        enable_gqa: bool,
    ) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.scale = scale
        self.enable_gqa = enable_gqa

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            torch.ops.aten.scaled_dot_product_attention.default(
                query,
                key,
                value,
                attention_mask,
                dropout_p=self.dropout_p,
                is_caual=self.is_causal,
                scale=self.scale,
                enable_gqa=self.enable_gqa,
            ),
        )
