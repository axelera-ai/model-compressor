# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from collections.abc import Set
from typing import Any

import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base

import model_compressor.passes.annotate.base as base
import model_compressor.passes.annotate.candidate_schemas as candidate_schemas
import model_compressor.passes.annotate.choose_schemas as choose_schemas
import model_compressor.passes.annotate.picker as _picker
import model_compressor.passes.annotate.source_schemas as source_schemas


class Annotate(pass_base.PassBase):
    def __init__(
        self,
        *,
        input: dict[str, Any],
        database: Set[base.SchemaProtocol],
        picker: _picker.Picker,
    ) -> None:
        self._input = input
        self._database = database
        self._picker = picker

    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        for node in graph_module.graph.nodes:
            annotation = base.Annotation(source=None, chosen=None)
            base.set_annotation(node, annotation)
        graph_module = source_schemas.source_schemas(graph_module, self._input)
        graph_module = choose_schemas.choose_schemas(
            graph_module,
            candidate_schemas.candidate_schemas(graph_module, self._input, self._database),
            self._picker,
        )
        return pass_base.PassResult(graph_module, True)
