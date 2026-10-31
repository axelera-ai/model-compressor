# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

"""Define the structural contract shared by all operator schemas."""

from collections.abc import Callable
from typing import Protocol, TypeAlias

import model_compressor.database.schema as _schema

DType: TypeAlias = _schema.DType
"""A catalogue of data types."""

ScalarDType: TypeAlias = _schema.ScalarDType
"""A catalogue of scalar data types."""

TensorDType: TypeAlias = _schema.TensorDType
"""A catalogue of tensor data types."""

Argument: TypeAlias = _schema.Argument
"""A name/type pair describing an operator's input or output."""

Signature: TypeAlias = _schema.Signature
"""A pair of ordered collections of name/type pairs describing an operator's I/O."""

ComputingEngine: TypeAlias = _schema.ComputingEngine
"""A catalogue of computing engines available on a hardware platform generation."""


class SchemaProtocol(Protocol):
    """The structural contract shared by all operator schemas - both JIT-validated and free-form.

    Attributes:
        engine: The computing engine on which the operator executes.
        target: The operator - an ATen or custom ``OpOverload`` or a builtin such as
            ``operator.getitem`` (hence the widest common type, ``Callable``).
        signature: The ordered name/type pair collections describing the operator's I/O.
        metadata: A Markdown string with implementation notes.

    """

    @property
    def engine(self) -> ComputingEngine:
        """Read-only: the computing engine on which the operator executes."""

    @property
    def target(self) -> Callable:
        """Read-only: the operator - an ATen or custom ``OpOverload`` or a builting such as operator.getitem``."""

    @property
    def signature(self) -> Signature:
        """Read-only: the ordered name/type pair collections describing the operator's I/O."""

    @property
    def metadata(self) -> str:
        """Read-only: a Markdown string with implementation notes."""


__all__ = ["DType", "ScalarDType", "TensorDType", "Argument", "Signature", "ComputingEngine", "SchemaProtocol"]
