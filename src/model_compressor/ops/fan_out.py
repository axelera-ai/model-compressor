# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch

import model_compressor.ops.namespace as namespace

_lib = torch.library.Library(namespace.NAMESPACE, "FRAGMENT")

_NAME: str = "fan_out"
"""The operator's name."""


# tensor overload
_OVERLOAD = "Tensor"
_lib.define(f"{_NAME}.{_OVERLOAD}(Tensor(a) input) -> Tensor(a)")
_lib.impl(f"{_NAME}.{_OVERLOAD}", lambda input: torch.ops.aten.alias.default(input), "CompositeExplicitAutograd")
_lib.impl(f"{_NAME}.{_OVERLOAD}", lambda input: torch.ops.aten.alias.default(input), "Meta")
_lib.impl(f"{_NAME}.{_OVERLOAD}", torch.library.fallthrough_kernel, "Autograd")


fan_out_tensor = getattr(getattr(getattr(torch.ops, namespace.NAMESPACE), _NAME), _OVERLOAD)
"""Reify a fan-out point.

Args:
    input: The tensor to be fanned-out.

Returns:
    The fanned-out tensor.

"""


# integer overload
_OVERLOAD = "int"
# The signature is ``(SymInt input) -> SymInt`` as opposed to ``(int input) -> int`` so that:
# * the operator can survive  re-``torch.export.export`` cycles with dynamic shapes;
# * it is compatible with non-dynamic integers.
_lib.define(f"{_NAME}.{_OVERLOAD}(SymInt input) -> SymInt")
_lib.impl(f"{_NAME}.{_OVERLOAD}", lambda input: input, "CompositeExplicitAutograd")
_lib.impl(f"{_NAME}.{_OVERLOAD}", lambda input: input, "Meta")


fan_out_int = getattr(getattr(getattr(torch.ops, namespace.NAMESPACE), _NAME), _OVERLOAD)
"""Reify a fan-out point.

Args:
    input: The integer to be fanned-out.

Returns:
    The fanned-out integer.

"""
