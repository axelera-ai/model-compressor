# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import dataclasses
from typing import cast

import torch.fx as fx

import model_compressor.passes.annotate.base.schema_protocol as _schema_protocol


@dataclasses.dataclass
class Annotation:
    """A source/chosen schema pair.

    Args:
        source: A source schema - can only be non-``None`` when annotating ``"call_function"`` nodes.
        chosen: A target schema - can only be non-``None`` when annotating ``"call_function"`` nodes.

    """

    source: None | _schema_protocol.SchemaProtocol
    chosen: None | _schema_protocol.SchemaProtocol


_KEY_META_ANNOTATION = "annotation"
"""The key under which a node's source/chosen schema pair is stored."""


def has_annotation(node: fx.Node) -> bool:
    """Verify whether a node has source/chosen schemas."""
    return _KEY_META_ANNOTATION in node.meta


def set_annotation(node: fx.Node, annotation: Annotation) -> None:
    """Write a node's source/chosen schemas."""
    node.meta[_KEY_META_ANNOTATION] = annotation


def get_annotation(node: fx.Node) -> Annotation:
    """Read a node's source/chosen schemas."""
    return cast(Annotation, node.meta[_KEY_META_ANNOTATION])


__all__ = ["Annotation", "has_annotation", "set_annotation", "get_annotation"]
