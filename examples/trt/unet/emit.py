# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

"""Emit strict_W.onnx / permissive_W.onnx for U-Net from the materialized placement.

Run from examples/model-compressor with
  PYTHONPATH=../packages/model-compressor/src:. python trt/unet/emit.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # trt/ -> emit_common

import emit_common
import models.unet as model

if __name__ == "__main__":
    emit_common.main(model.create_unet, os.path.dirname(os.path.abspath(__file__)))
