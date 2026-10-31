# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

"""Define the free-form model compressor (MC) operator schema.

Where ``database.Schema`` is the JIT-validated description of an operator used to compile
hardware-generation-specific operator databases, a free-form MC operator schema can be assembled
declaratively wherever a schema-like description is needed. A free-form MC schema is lightweight
in that it does not perofrm any JIT schema validation.

"""

import dataclasses
from collections.abc import Callable

import model_compressor.passes.annotate.base.schema_protocol as _schema_protocol


@dataclasses.dataclass(frozen=True)
class Schema:
    """A free-form MC operator schema.

    Args:
        engine: The computing engine where the operator executes.
        target: The operator - an ATen or custom ``OpOverload`` or a builtin such as
            ``operator.getitem`` (hence the widest common type, ``Callable``).
        signature: The ordered name/type pair collections describing the operator's I/O.
        metadata: A Markdown string with implementation notes.

    """

    engine: _schema_protocol.ComputingEngine
    target: Callable
    signature: _schema_protocol.Signature
    metadata: str


__all__ = ["Schema"]
