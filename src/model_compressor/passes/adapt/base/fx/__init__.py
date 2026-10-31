# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from model_compressor.passes.adapt.base.fx.semantics import Semantics
from model_compressor.passes.adapt.base.fx.subgraph import (
    KWArgToSubgraph,
    Subgraph,
    get_kwarg_to_subgraph,
    has_kwarg_to_subgraph,
    set_kwarg_to_subgraph,
)
