# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch

import model_compressor.ops.namespace as namespace


@torch.library.custom_op(f"{namespace.NAMESPACE}::recode", mutates_args=())
def recode(input: torch.Tensor) -> torch.Tensor:
    """Recode (decompress-and-compress) a tensor into from a latent representation into another.

    Args:
        input: The tensor to recode.

    Returns:
        The recoded tensor.

    """
    return input.clone()


@recode.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input)
