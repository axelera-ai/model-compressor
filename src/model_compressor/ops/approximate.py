# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch

import model_compressor.ops.namespace as namespace


@torch.library.custom_op(f"{namespace.NAMESPACE}::approximate", mutates_args=())
def approximate(input: torch.Tensor) -> torch.Tensor:
    """Approximate a non-linear operator acting elementwise on a tensor.

    Args:
        input: A tensor.

    Returns:
        The approximated non-linearly transformed tensor.

    """
    return input.clone()


@approximate.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input)
