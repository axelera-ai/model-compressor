# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import dataclasses
from collections.abc import Set
from typing import Any

import model_compressor.api.catalog as catalog
import model_compressor.api.config as config
import model_compressor.database as database
import model_compressor.nn as nn
import model_compressor.passes as passes

CATALOG = catalog.Catalog(database.generations.alpha.DATABASE)
"""The catalogue of operators on the Alpha hardware platform generation."""


X_PICKER = passes.annotate.PriorityPicker(
    [
        # SI8 matmuls
        CATALOG.engine_aten_convolution_si8si8nulllintlintlintboollintint_si32,
        CATALOG.engine_aten_linear_si8si8null_si32,
        CATALOG.engine_aten_scaled_dot_product_attention_si8si8si8f32floatboolnullbool_si32,
        CATALOG.engine_aten_scaled_dot_product_attention_si8si8si8f32floatboolfloatbool_si32,
        # SI8 routing
        CATALOG.engine_aten_cat_lsi8int_si8,
        CATALOG.engine_aten_split_with_sizes_si8lintint_lsi8,
        CATALOG.engine_aten_sym_size_int_si8int_int,
        CATALOG.engine_xaten_fan_out_Tensor_si8_si8,
        # F32
        CATALOG.engine_aten_clone_f32int_f32,
        CATALOG.engine_aten_clone_f32null_f32,
        CATALOG.engine_aten_constant_pad_nd_f32lintnumber_f32,
        CATALOG.engine_aten_expand_f32lintbool_f32,
        CATALOG.engine_aten_max_pool2d_with_indices_f32lintlintlintlintbool_f32si64,
        CATALOG.engine_aten_permute_f32lint_f32,
        CATALOG.engine_aten_slice_Tensor_f32intintintint_f32,
        CATALOG.engine_aten_upsample_nearest2d_vec_f32lintnull_f32,
        CATALOG.engine_aten_view_f32lint_f32,
        CATALOG.engine_xaten_kv_cache_f32f32intint_f32f32,
    ],
)
"""Policy X: SI8 matmuls and routing (fan-out, split, concat); F32 elsewhere."""

Y_PICKER = passes.annotate.PriorityPicker(
    [
        # SI8 matmuls
        CATALOG.engine_aten_convolution_si8si8nulllintlintlintboollintint_si32,
        CATALOG.engine_aten_linear_si8si8null_si32,
        CATALOG.engine_aten_scaled_dot_product_attention_si8si8si8f32floatboolnullbool_si32,
        CATALOG.engine_aten_scaled_dot_product_attention_si8si8si8f32floatboolfloatbool_si32,
        # F32 routing
        CATALOG.engine_aten_cat_lf32int_f32,
        CATALOG.engine_aten_split_with_sizes_f32lintint_lf32,
        CATALOG.engine_aten_sym_size_int_f32int_int,
        CATALOG.engine_xaten_fan_out_Tensor_f32_f32,
        # F32
        CATALOG.engine_aten_clone_f32int_f32,
        CATALOG.engine_aten_clone_f32null_f32,
        CATALOG.engine_aten_constant_pad_nd_f32lintnumber_f32,
        CATALOG.engine_aten_expand_f32lintbool_f32,
        CATALOG.engine_aten_max_pool2d_with_indices_f32lintlintlintlintbool_f32si64,
        CATALOG.engine_aten_permute_f32lint_f32,
        CATALOG.engine_aten_slice_Tensor_f32intintintint_f32,
        CATALOG.engine_aten_upsample_nearest2d_vec_f32lintnull_f32,
        CATALOG.engine_aten_view_f32lint_f32,
        CATALOG.engine_xaten_kv_cache_f32f32intint_f32f32,
    ],
)
"""Policy Y: SI8 matmuls; F32 routing (fan-out, split, concat); F32 elsewhere."""

Z_PICKER = passes.annotate.PriorityPicker(
    [
        # F32 matmuls
        CATALOG.engine_aten_convolution_f32f32nulllintlintlintboollintint_f32,
        CATALOG.engine_aten_linear_f32f32null_f32,
        CATALOG.engine_aten_scaled_dot_product_attention_f32f32f32f32floatboolnullbool_f32,
        CATALOG.engine_aten_scaled_dot_product_attention_f32f32f32f32floatboolfloatbool_f32,
        # F32 routing
        CATALOG.engine_aten_cat_lf32int_f32,
        CATALOG.engine_aten_split_with_sizes_f32lintint_lf32,
        CATALOG.engine_aten_sym_size_int_f32int_int,
        CATALOG.engine_xaten_fan_out_Tensor_f32_f32,
        # F32
        CATALOG.engine_aten_clone_f32int_f32,
        CATALOG.engine_aten_clone_f32null_f32,
        CATALOG.engine_aten_constant_pad_nd_f32lintnumber_f32,
        CATALOG.engine_aten_expand_f32lintbool_f32,
        CATALOG.engine_aten_max_pool2d_with_indices_f32lintlintlintlintbool_f32si64,
        CATALOG.engine_aten_permute_f32lint_f32,
        CATALOG.engine_aten_slice_Tensor_f32intintintint_f32,
        CATALOG.engine_aten_upsample_nearest2d_vec_f32lintnull_f32,
        CATALOG.engine_aten_view_f32lint_f32,
        CATALOG.engine_xaten_kv_cache_f32f32intint_f32f32,
    ],
)
"""Policy Z: unquantized F32 reference."""

W_PICKER = passes.annotate.PriorityPicker(
    [
        # SI8 matmuls
        CATALOG.engine_aten_convolution_si8si8nulllintlintlintboollintint_si32,
        CATALOG.engine_aten_linear_si8si8null_si32,
        CATALOG.engine_aten_scaled_dot_product_attention_si8si8si8f32floatboolnullbool_si32,
        CATALOG.engine_aten_scaled_dot_product_attention_si8si8si8f32floatboolfloatbool_si32,
        # SI8 routing
        CATALOG.engine_aten_cat_lsi8int_si8,
        CATALOG.engine_aten_split_with_sizes_si8lintint_lsi8,
        CATALOG.engine_aten_sym_size_int_si8int_int,
        CATALOG.engine_xaten_fan_out_Tensor_si8_si8,
        # SI8 everywhere else
        CATALOG.engine_aten_clone_si8int_si8,
        CATALOG.engine_aten_clone_si8null_si8,
        CATALOG.engine_aten_constant_pad_nd_si8lintnumber_si8,
        CATALOG.engine_aten_expand_si8lintbool_si8,
        CATALOG.engine_aten_max_pool2d_with_indices_si8lintlintlintlintbool_si8si64,
        CATALOG.engine_aten_permute_si8lint_si8,
        CATALOG.engine_aten_slice_Tensor_si8intintintint_si8,
        CATALOG.engine_aten_upsample_nearest2d_vec_si8lintnull_si8,
        CATALOG.engine_aten_view_si8lint_si8,
        CATALOG.engine_xaten_kv_cache_si8si8intint_si8si8,
    ],
)
"""Policy W: SI8 aggressively - every operator with an SI8 variant is kept in SI8."""


_OPTIMIZING_MODULE_RULES = [
    config.OptimizingModuleRule(
        module=nn.OptimizingActivation,
        modularize=passes.modularize.ModularizeIntoOptimizingActivation,
        functionalize=passes.functionalize.FunctionalizeFromOptimizingActivation,
    ),
    config.OptimizingModuleRule(
        module=nn.OptimizingBatchNorm,
        modularize=passes.modularize.ModularizeIntoOptimizingBatchNorm,
        functionalize=passes.functionalize.FunctionalizeFromOptimizingBatchNorm,
    ),
    config.OptimizingModuleRule(
        module=nn.OptimizingConvolution,
        modularize=passes.modularize.ModularizeIntoOptimizingConvolution,
        functionalize=passes.functionalize.FunctionalizeFromOptimizingConvolution,
    ),
    config.OptimizingModuleRule(
        module=nn.OptimizingLinear,
        modularize=passes.modularize.ModularizeIntoOptimizingLinear,
        functionalize=passes.functionalize.FunctionalizeFromOptimizingLinear,
    ),
    config.OptimizingModuleRule(
        module=nn.OptimizingRMSNorm,
        modularize=passes.modularize.ModularizeIntoOptimizingRMSNorm,
        functionalize=passes.functionalize.FunctionalizeFromOptimizingRMSNorm,
    ),
    config.OptimizingModuleRule(
        module=nn.OptimizingSDPA,
        modularize=passes.modularize.ModularizeIntoOptimizingSDPA,
        functionalize=passes.functionalize.FunctionalizeFromOptimizingSDPA,
    ),
]


@dataclasses.dataclass
class _AnnotationConfig(config.AnnotationConfig):
    database: Set[passes.annotate.SchemaProtocol] = dataclasses.field(
        default_factory=lambda: database.generations.alpha.DATABASE, init=False
    )
    picker: passes.annotate.Picker = dataclasses.field(default=lambda _, candidates: candidates.pop())
    input: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _AdaptationConfig(config.AdaptationConfig):
    assignment: passes.adapt.AssignmentMode = dataclasses.field(default=passes.adapt.AssignmentMode.PERMISSIVE)
    coalescence: passes.adapt.CoalescenceMode = dataclasses.field(default=passes.adapt.CoalescenceMode.WITH)


@dataclasses.dataclass
class _OptimizationConfig(config.OptimizationConfig):
    rules: list[config.OptimizingModuleRule] = dataclasses.field(
        default_factory=lambda: _OPTIMIZING_MODULE_RULES, init=False
    )


@dataclasses.dataclass
class Config(config.Config):
    annotation: config.AnnotationConfig = dataclasses.field(default_factory=lambda: _AnnotationConfig(), init=False)
    adaptation: config.AdaptationConfig = dataclasses.field(default_factory=lambda: _AdaptationConfig(), init=False)
    optimization: config.OptimizationConfig = dataclasses.field(
        default_factory=lambda: _OptimizationConfig(), init=False
    )
