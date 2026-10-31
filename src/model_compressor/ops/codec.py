# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch

import model_compressor.ops.namespace as namespace


@torch.library.custom_op(f"{namespace.NAMESPACE}::codec", mutates_args=())
def codec(input: torch.Tensor) -> torch.Tensor:
    """Encode-and-decode a tensor into-and-from its latent representation.

    Args:
        input: The tensor to encode-and-decode.

    Returns:
        The encoded-and-decoded tensor.

    """
    return input.clone()


@codec.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input)
