# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import Any

import segmentation_models_pytorch as smp
import torch
import torch.fx as fx
from models.common import export


def create_unet() -> tuple[fx.GraphModule, dict[str, Any]]:
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    return export(model, {"x": torch.randn(1, 3, 512, 512)})
