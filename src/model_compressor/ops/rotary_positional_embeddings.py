# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import Any

import torch

import model_compressor.ops.namespace as namespace


@torch.library.custom_op(f"{namespace.NAMESPACE}::rotary_positional_embeddings", mutates_args=())
def rotary_positional_embeddings(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate positional embeddings.

    Similar to Hugging Face's `apply_rotary_pos_emb` function, but this one can be
    applied on the keys and queries separately.

    Note: in contrast to the Hugging Face implementation, our cos and sin inputs were not
    repeated along the last dimension (head_dim), so we have to apply the concatenation here.

    Note: unlike HF (which passes 4D trig tensors with a singleton head dimension), our backbone
    passes 3D trig tensors of shape (batch, seq_len, head_dim//2) without the head dimension.
    The missing head dimension is inserted internally before broadcasting with x.

    Args:
        x: The tensor to be rotated. Shape: (batch, heads, seq_len, head_dim) - last dimension must be even.
        cos: The cosine of the rotation angle. Shape: (batch, seq_len, head_dim//2).
        sin: The sine of the rotation angle. Shape: (batch, seq_len, head_dim//2).

    Returns:
        The rotated tensor.

    """
    expected_trig_shape = (x.shape[0], x.shape[2], x.shape[3] // 2)
    if cos.shape != expected_trig_shape or sin.shape != expected_trig_shape:
        raise ValueError(
            f"Expected cos and sin to have 3D shape {expected_trig_shape} "
            f"(batch, seq_len, head_dim//2) for x of shape {tuple(x.shape)}, "
            f"but got cos shape {tuple(cos.shape)} and sin shape {tuple(sin.shape)}."
        )
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x1, x2 = x.chunk(2, dim=-1)
    x_embed = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return x_embed.contiguous()


@rotary_positional_embeddings.register_fake
def _(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x).contiguous()


def rotary_positional_embeddings_setup_context(ctx, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
    x, cos, sin = inputs
    ctx.save_for_backward(x, cos, sin)


def rotary_positional_embeddings_backward(
    ctx, grad_output: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, cos, sin = ctx.saved_tensors
    cos_4d = cos.unsqueeze(1)
    sin_4d = sin.unsqueeze(1)
    g1, g2 = grad_output.chunk(2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)

    # Forward: out = cat(x1*cos - x2*sin, x2*cos + x1*sin), with cos/sin broadcast over heads.
    grad_x = torch.cat((g1 * cos_4d + g2 * sin_4d, -g1 * sin_4d + g2 * cos_4d), dim=-1)
    # cos/sin are 3D and broadcast across the heads dim, so reduce that dim back out.
    grad_cos = (g1 * x1 + g2 * x2).sum(dim=1)
    grad_sin = (-g1 * x2 + g2 * x1).sum(dim=1)

    return grad_x, grad_cos, grad_sin


rotary_positional_embeddings.register_autograd(
    rotary_positional_embeddings_backward, setup_context=rotary_positional_embeddings_setup_context
)
