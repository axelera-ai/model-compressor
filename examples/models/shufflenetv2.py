# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import Any

import torch
import torch.fx as fx
import torchvision
from models.common import export


def create_shufflenetv2() -> tuple[fx.GraphModule, dict[str, Any]]:
    model = torchvision.models.shufflenet_v2_x1_0(weights=torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    return export(model, {"x": torch.randn((1, 3, 224, 224))})
