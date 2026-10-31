# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from model_compressor.passes.modularize.modularize_into_optimizing_activation import (
    ModularizeIntoOptimizingActivation,
)
from model_compressor.passes.modularize.modularize_into_optimizing_batch_norm import (
    ModularizeIntoOptimizingBatchNorm,
)
from model_compressor.passes.modularize.modularize_into_optimizing_convolution import (
    ModularizeIntoOptimizingConvolution,
)
from model_compressor.passes.modularize.modularize_into_optimizing_linear import (
    ModularizeIntoOptimizingLinear,
)
from model_compressor.passes.modularize.modularize_into_optimizing_rms_norm import (
    ModularizeIntoOptimizingRMSNorm,
)
from model_compressor.passes.modularize.modularize_into_optimizing_sdpa import ModularizeIntoOptimizingSDPA
