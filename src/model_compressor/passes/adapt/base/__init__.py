# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from model_compressor.passes.adapt.base import fx, nx
from model_compressor.passes.adapt.base.convert import convert_fx_to_nx, convert_nx_to_fx
from model_compressor.passes.adapt.base.convert_subgraph import (
    convert_fx_subgraphs_to_nx_subgraphs,
    convert_nx_subgraphs_to_fx_subgraphs,
)
