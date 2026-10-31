# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import Protocol

import torch

import model_compressor.ops.namespace as namespace


class _KVCacheT(Protocol):
    """The interface for key-value (KV) caches."""

    def __len__(self) -> int:
        """The number of attention layers for which the cache is hold."""

    def get_seq_length(self, layer_idx: int) -> int:
        """The length of cached KV token sequences for a given attention layer.

        Args:
            layer_idx: The index of the attention layer.

        """

    def update(
        self, key: torch.Tensor, value: torch.Tensor, layer_idx: int, total_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the KV cache of a given attention layer.

        Args:
            key: The current key (K) tensor.
            value: The current value (V) tensor.
            layer_idx: The index of the attention layer.
            total_seq_len: The total length of KV token sequences for the attention layer.

        Returns:
            A pair whose first item is the complete key tensor (cached and non-cached parts), and
                whose second item is the complete value tensor (cached and non-cached parts).

        """


class NoCache:
    """A dummy KV cache - not performing any caching."""

    def __len__(self) -> int:
        """The number of attention layers for which the cache is hold."""
        return 0

    def get_seq_length(self, layer_idx: int) -> int:
        """The length of cached KV token sequences for a given attention layer.

        Args:
            layer_idx: The index of the attention layer.

        """
        return 0

    def update(
        self, key: torch.Tensor, value: torch.Tensor, layer_idx: int, total_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the KV cache of a given attention layer.

        Args:
            key: The current key (K) tensor.
            value: The current value (V) tensor.
            layer_idx: The index of the attention layer.
            total_seq_len: The total length of KV token sequences for the attention layer.

        Returns:
            A pair whose first item is the complete key tensor (cached and non-cached parts), and
                whose second item is the complete value tensor (cached and non-cached parts).

        """
        return key, value


_KV_CACHE: _KVCacheT = NoCache()
"""A dummy KV cache."""


def get_kv_cache() -> _KVCacheT:
    """Get the KV cache.

    Returns:
        The KV cache used.

    """
    global _KV_CACHE  # noqa: F824
    return _KV_CACHE


def set_kv_cache(kv_cache: _KVCacheT) -> None:
    """Set the KV cache.

    Args:
        kv_cache: The KV cache to use.

    """
    global _KV_CACHE
    _KV_CACHE = kv_cache


@torch.library.custom_op(f"{namespace.NAMESPACE}::kv_cache", mutates_args=())
def kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_idx: int,
    total_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update and retrieve the K and V tensors from the KV cache for a given attention layer.

    Args:
        key: The key tensor (K). Shape: (batch_size, num_heads, seq_len, head_dim).
        value: The value tensor (V). Shape: (batch_size, num_heads, seq_len, head_dim).
        layer_idx: The index of the attention layer for which the KV cache is being updated.
        total_seq_len: The total sequence length for the K and V tensors - i.e., including the
            cached states.

    Returns:
        The updated key (K) and value (V) tensors.

    Raises:
        ValueError: The key or value tensor does not have the expected sequence length.

    """
    all_key, all_value = _KV_CACHE.update(key, value, layer_idx, total_seq_len)
    if (id(all_key) == id(key)) or (id(all_value) == id(value)):
        # avoid tracing errors due to in-place processing
        all_key, all_value = key.clone(), value.clone()
    if (all_key.size(2) != total_seq_len) or (all_value.size(2) != total_seq_len):
        raise ValueError(
            f"Key and value tensors must have the same size as the total sequence length. "
            f"Got key size {all_key.size(2)} and value size {all_value.size(2)}, but expected "
            f"{total_seq_len}. This indicates a problem with the KV cache management."
        )
    return all_key.contiguous(), all_value.contiguous()


@kv_cache.register_fake
def _(key: torch.Tensor, value: torch.Tensor, layer_idx: int, total_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_heads, seq_len, head_dim = key.shape
    all_key = torch.empty((batch_size, num_heads, total_seq_len, head_dim), dtype=key.dtype, device=key.device)
    all_value = torch.empty((batch_size, num_heads, total_seq_len, head_dim), dtype=value.dtype, device=value.device)
    return all_key.contiguous(), all_value.contiguous()


def kv_cache_setup_context(ctx, inputs: tuple[torch.Tensor, torch.Tensor, int, int], output: torch.Tensor) -> None:
    key, value, layer_idx, total_seq_len = inputs
    ctx.save_for_backward(key, value)


def kv_cache_backward(
    ctx, grad_key: torch.Tensor, grad_value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, None, None]:
    key, _ = ctx.saved_tensors
    seq_len = key.size(2)
    return grad_key[:, :, -seq_len:, :], grad_value[:, :, -seq_len:, :], None, None


kv_cache.register_autograd(kv_cache_backward, setup_context=kv_cache_setup_context)
