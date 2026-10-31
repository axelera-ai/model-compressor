# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from model_compressor.passes.canonicalize.canonicalize_bias import CanonicalizeBias
from model_compressor.passes.canonicalize.canonicalize_convolution import CanonicalizeConvolution
from model_compressor.passes.canonicalize.canonicalize_linear import CanonicalizeLinear
from model_compressor.passes.canonicalize.canonicalize_max_poolnd_with_indices import (
    CanonicalizeMaxPool2dWithIndices,
    CanonicalizeMaxPool3dWithIndices,
)
from model_compressor.passes.canonicalize.reify_fan_out import ReifyFanOut

__all__ = [
    "CanonicalizeBias",
    "CanonicalizeConvolution",
    "CanonicalizeLinear",
    "CanonicalizeMaxPool2dWithIndices",
    "CanonicalizeMaxPool3dWithIndices",
    "ReifyFanOut",
]
