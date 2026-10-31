# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

"""Emit strict_W.onnx / permissive_W.onnx for SigLIP (SmolVLM2 vision tower) from the
materialized placement.

Run from examples/model-compressor with
  PYTHONPATH=../packages/model-compressor/src:. python trt/siglip/emit.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # trt/ -> emit_common

import emit_common
import models.smolvlm2 as model

if __name__ == "__main__":
    emit_common.main(model.create_smolvlm2_vision, os.path.dirname(os.path.abspath(__file__)))
