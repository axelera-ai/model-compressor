# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

"""Routing fixture -- a synthetic model to exercise coalescence of routing (concat) operators.

The backbone builds a balanced binary **tree of** ``torch.cat`` **ops**:
four leaf inputs are joined pairwise, then the two partial results are
joined at the root.

``aten.cat`` is the one op here whose chosen Alpha schema is ``SI8`` on
*both* its inputs and its output, so a ``cat`` feeding a ``cat`` places
two ``fake`` (F32 -> SI8) domains right next to each other. In ``strict``
mode the inserted codecs split the tree into several compatible fake
domains; coalescence then merges the adjacent ones back (``exits <=
boundary`` holds -- no branch escapes to the model output).

"""

from typing import Any

import torch
import torch.fx as fx
from models.common import export

DIM = 4


class _Routing(torch.nn.Module):
    def forward(self, x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        c01 = torch.cat([x0, x1], dim=-1)  # (1, 2 * DIM)
        c23 = torch.cat([x2, x3], dim=-1)  # (1, 2 * DIM)
        return torch.cat([c01, c23], dim=-1)  # (1, 4 * DIM) -- the root of the cat tree


def create_routing() -> tuple[fx.GraphModule, dict[str, Any]]:
    return export(_Routing(), {f"x{i}": torch.randn((1, DIM)) for i in range(4)})
