# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch

import model_compressor.ops.namespace as namespace


@torch.library.custom_op(f"{namespace.NAMESPACE}::encode", mutates_args=())
def encode(input: torch.Tensor) -> torch.Tensor:
    """Encode (compress) a tensor into its latent representation.

    Args:
        input: The tensor to encode.

    Returns:
        The encoded tensor.

    """
    return input.clone()


@encode.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input)
