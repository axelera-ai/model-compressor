# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import Any

import torch
import torch.fx as fx
import torchvision
from models.common import export


def create_resnet18() -> tuple[fx.GraphModule, dict[str, Any]]:
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    return export(model, {"x": torch.randn((1, 3, 224, 224))})
