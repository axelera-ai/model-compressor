# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch

import model_compressor.ops.namespace as namespace


@torch.library.custom_op(f"{namespace.NAMESPACE}::decode", mutates_args=())
def decode(input: torch.Tensor) -> torch.Tensor:
    """Decode (decompress) a tensor from its latent representation.

    Args:
        input: The tensor to decode.

    Returns:
        The decoded tensor.

    """
    return input.clone()


@decode.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input)
