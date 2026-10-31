# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import dataclasses
from collections.abc import Callable
from typing import Any, Literal, cast

import torch
import torch.nn as nn

import model_compressor.passes.adapt.base.nx.graph as _graph
import model_compressor.passes.annotate.base as annotate


@dataclasses.dataclass(frozen=True)
class Annotation:
    """A description of an ``fx.Node`` when represented in a ``networkx.MultiDiGraph``.

    Args:
        op: The FX opcode.
        target: The FX target.
        value: For `"get_attr"` nodes, the value of the associated buffer or parameter.
        kwargs: A copy of the keyword arguments where (possibly nested ) ``fx.Node`` items have been stubbed with
              sentinel values; the ``fx.Node`` arguments are captured by the ``networkx.MultiDiGraph``'s connectivity.
        annotation: For `"call_function"` nodes, the descriptions of the operator in the source and optimized models.

    """

    op: Literal["placeholder", "get_attr", "output", "call_function"]
    target: str | Callable
    value: None | torch.Tensor | nn.Parameter
    kwargs: dict[str, Any]
    annotation: annotate.Annotation

    @property
    def source(self) -> annotate.SchemaProtocol:
        """For ``"call_function"`` nodes, the description of the operator in the source model."""
        return cast(annotate.SchemaProtocol, self.annotation.source)

    @property
    def chosen(self) -> annotate.SchemaProtocol:
        """For ``"call_function"`` nodes, the description of the operator in the optimized model."""
        return cast(annotate.SchemaProtocol, self.annotation.chosen)


_KEY_NODE_ANNOTATION: str = "annotation"
"""The key under which a node's annotation is stored."""


def has_annotation(graph: _graph.Graph, node: _graph.Node) -> bool:
    """Verify whether a node has an annotation."""
    return _KEY_NODE_ANNOTATION in graph.nodes[node]


def set_annotation(graph: _graph.Graph, node: _graph.Node, annotation: Annotation) -> None:
    """Write a node's annotation."""
    graph.nodes[node][_KEY_NODE_ANNOTATION] = annotation


def get_annotation(graph: _graph.Graph, node: _graph.Node) -> Annotation:
    """Read a node's annotation."""
    return cast(Annotation, graph.nodes[node][_KEY_NODE_ANNOTATION])
