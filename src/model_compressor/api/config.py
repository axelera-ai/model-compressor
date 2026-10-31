# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import dataclasses
from collections.abc import Set
from typing import Any

import torch.fx.passes.infra.pass_base as pass_base
import torch.nn as nn

import model_compressor.passes as passes


@dataclasses.dataclass
class AnnotationConfig:
    database: Set[passes.annotate.SchemaProtocol]
    picker: passes.annotate.Picker
    input: dict[str, Any]


@dataclasses.dataclass
class AdaptationConfig:
    assignment: passes.adapt.AssignmentMode
    coalescence: passes.adapt.CoalescenceMode


@dataclasses.dataclass
class OptimizingModuleRule:
    module: type[nn.Module]
    modularize: type[pass_base.PassBase]
    functionalize: type[pass_base.PassBase]


@dataclasses.dataclass
class OptimizationConfig:
    rules: list[OptimizingModuleRule]


@dataclasses.dataclass
class Config:
    annotation: AnnotationConfig
    adaptation: AdaptationConfig
    optimization: OptimizationConfig
