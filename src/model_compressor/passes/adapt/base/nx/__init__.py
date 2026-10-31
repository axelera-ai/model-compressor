# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from model_compressor.passes.adapt.base.nx.annotation import (
    Annotation,
    get_annotation,
    has_annotation,
    set_annotation,
)
from model_compressor.passes.adapt.base.nx.graph import (
    Edge,
    Graph,
    Node,
    PassResult,
    Port,
)
from model_compressor.passes.adapt.base.nx.semantics import (
    Semantics,
    get_input_port_semantics,
    get_output_port_semantics,
    get_semantics,
    has_semantics,
    set_semantics,
)
from model_compressor.passes.adapt.base.nx.subgraph import (
    Subgraph,
    get_subgraph,
    has_subgraph,
    set_subgraph,
)
