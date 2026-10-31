# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

"""Define the database of operators for the Alpha hardware platform generation."""

import enum
from collections.abc import Set
from typing import Literal

import torch

import model_compressor.database.schema as schema
import model_compressor.ops  # noqa: F401 - register extended ATen operators


class _Alpha(schema.ComputingEngine):
    """The computing engines of the Alpha hardware platform generation."""

    ENGINE = enum.auto()


_ENGINE: Set[schema.Schema] = frozenset(
    {
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten._native_batch_norm_legit_no_training.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="input", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="weight", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="bias", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="running_mean", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="running_var", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="momentum", type=Literal[schema.ScalarDType.FLOAT]),
                    schema.Argument(name="eps", type=Literal[schema.ScalarDType.FLOAT]),
                ],
                output=[
                    schema.Argument(name="", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="", type=Literal[schema.TensorDType.F32]),
                ],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.add.Tensor,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="other", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="alpha", type=Literal[schema.ScalarDType.NUMBER]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.cat.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="tensors", type=list[Literal[schema.TensorDType.SI8]]),
                    schema.Argument(name="dim", type=Literal[schema.ScalarDType.INT]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.SI8])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.cat.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="tensors", type=list[Literal[schema.TensorDType.F32]]),
                    schema.Argument(name="dim", type=Literal[schema.ScalarDType.INT]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.clone.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="memory_format", type=Literal[schema.ScalarDType.NULL]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.SI8])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.clone.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="memory_format", type=Literal[schema.ScalarDType.NULL]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.clone.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="memory_format", type=Literal[schema.ScalarDType.INT]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.SI8])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.clone.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="memory_format", type=Literal[schema.ScalarDType.INT]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.constant_pad_nd.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="pad", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="value", type=Literal[schema.ScalarDType.NUMBER]),
                ],
                output=[
                    schema.Argument(name="", type=Literal[schema.TensorDType.SI8]),
                ],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.constant_pad_nd.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="pad", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="value", type=Literal[schema.ScalarDType.NUMBER]),
                ],
                output=[
                    schema.Argument(name="", type=Literal[schema.TensorDType.F32]),
                ],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.convolution.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="input", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="weight", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="bias", type=Literal[schema.ScalarDType.NULL]),
                    schema.Argument(name="stride", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="padding", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="dilation", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="transposed", type=Literal[schema.ScalarDType.BOOL]),
                    schema.Argument(name="output_padding", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="groups", type=Literal[schema.ScalarDType.INT]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.SI32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.convolution.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="input", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="weight", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="bias", type=Literal[schema.ScalarDType.NULL]),
                    schema.Argument(name="stride", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="padding", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="dilation", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="transposed", type=Literal[schema.ScalarDType.BOOL]),
                    schema.Argument(name="output_padding", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="groups", type=Literal[schema.ScalarDType.INT]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.embedding.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="weight", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="indices", type=Literal[schema.TensorDType.SI64]),
                    schema.Argument(name="padding_idx", type=Literal[schema.ScalarDType.INT]),
                    schema.Argument(name="scale_grad_by_freq", type=Literal[schema.ScalarDType.BOOL]),
                    schema.Argument(name="sparse", type=Literal[schema.ScalarDType.BOOL]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.expand.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="size", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="implicit", type=Literal[schema.ScalarDType.BOOL]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.SI8])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.expand.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="size", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="implicit", type=Literal[schema.ScalarDType.BOOL]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.gelu.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="approximate", type=Literal[schema.ScalarDType.STRING]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.layer_norm.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="input", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="normalized_shape", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="weight", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="bias", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="eps", type=Literal[schema.ScalarDType.FLOAT]),
                    schema.Argument(name="cudnn_enable", type=Literal[schema.ScalarDType.BOOL]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.linear.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="input", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="weight", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="bias", type=Literal[schema.ScalarDType.NULL]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.SI32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.linear.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="input", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="weight", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="bias", type=Literal[schema.ScalarDType.NULL]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.max_pool2d_with_indices.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="kernel_size", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="stride", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="padding", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="dilation", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="ceil_mode", type=Literal[schema.ScalarDType.BOOL]),
                ],
                output=[
                    schema.Argument(name="", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="", type=Literal[schema.TensorDType.SI64]),
                ],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.max_pool2d_with_indices.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="kernel_size", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="stride", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="padding", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="dilation", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="ceil_mode", type=Literal[schema.ScalarDType.BOOL]),
                ],
                output=[
                    schema.Argument(name="", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="", type=Literal[schema.TensorDType.SI64]),
                ],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.mean.dim,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="dim", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="keepdim", type=Literal[schema.ScalarDType.BOOL]),
                    schema.Argument(name="dtype", type=Literal[schema.ScalarDType.NULL]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.mul.Tensor,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="other", type=Literal[schema.TensorDType.F32]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.permute.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="dims", type=list[Literal[schema.ScalarDType.INT]]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.SI8])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.permute.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="dims", type=list[Literal[schema.ScalarDType.INT]]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.relu.default,
            signature=schema.Signature(
                input=[schema.Argument(name="self", type=Literal[schema.TensorDType.F32])],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.rms_norm.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="input", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="normalized_shape", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="weight", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="eps", type=Literal[schema.ScalarDType.FLOAT]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.scaled_dot_product_attention.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="query", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="key", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="value", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="attn_mask", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="dropout_p", type=Literal[schema.ScalarDType.FLOAT]),
                    schema.Argument(name="is_causal", type=Literal[schema.ScalarDType.BOOL]),
                    schema.Argument(name="scale", type=Literal[schema.ScalarDType.NULL]),
                    schema.Argument(name="enable_gqa", type=Literal[schema.ScalarDType.BOOL]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.SI32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.scaled_dot_product_attention.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="query", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="key", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="value", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="attn_mask", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="dropout_p", type=Literal[schema.ScalarDType.FLOAT]),
                    schema.Argument(name="is_causal", type=Literal[schema.ScalarDType.BOOL]),
                    schema.Argument(name="scale", type=Literal[schema.ScalarDType.NULL]),
                    schema.Argument(name="enable_gqa", type=Literal[schema.ScalarDType.BOOL]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.scaled_dot_product_attention.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="query", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="key", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="value", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="attn_mask", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="dropout_p", type=Literal[schema.ScalarDType.FLOAT]),
                    schema.Argument(name="is_causal", type=Literal[schema.ScalarDType.BOOL]),
                    schema.Argument(name="scale", type=Literal[schema.ScalarDType.FLOAT]),
                    schema.Argument(name="enable_gqa", type=Literal[schema.ScalarDType.BOOL]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.SI32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.scaled_dot_product_attention.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="query", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="key", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="value", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="attn_mask", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="dropout_p", type=Literal[schema.ScalarDType.FLOAT]),
                    schema.Argument(name="is_causal", type=Literal[schema.ScalarDType.BOOL]),
                    schema.Argument(name="scale", type=Literal[schema.ScalarDType.FLOAT]),
                    schema.Argument(name="enable_gqa", type=Literal[schema.ScalarDType.BOOL]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.silu.default,
            signature=schema.Signature(
                input=[schema.Argument(name="self", type=Literal[schema.TensorDType.F32])],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.slice.Tensor,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="dim", type=Literal[schema.ScalarDType.INT]),
                    schema.Argument(name="start", type=Literal[schema.ScalarDType.INT]),
                    schema.Argument(name="end", type=Literal[schema.ScalarDType.INT]),
                    schema.Argument(name="step", type=Literal[schema.ScalarDType.INT]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.SI8])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.slice.Tensor,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="dim", type=Literal[schema.ScalarDType.INT]),
                    schema.Argument(name="start", type=Literal[schema.ScalarDType.INT]),
                    schema.Argument(name="end", type=Literal[schema.ScalarDType.INT]),
                    schema.Argument(name="step", type=Literal[schema.ScalarDType.INT]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.split_with_sizes.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="split_sizes", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="dim", type=Literal[schema.ScalarDType.INT]),
                ],
                output=[
                    schema.Argument(name="", type=list[Literal[schema.TensorDType.SI8]]),
                ],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.split_with_sizes.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="split_sizes", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="dim", type=Literal[schema.ScalarDType.INT]),
                ],
                output=[
                    schema.Argument(name="", type=list[Literal[schema.TensorDType.F32]]),
                ],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.sym_size.int,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="dim", type=Literal[schema.ScalarDType.INT]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.ScalarDType.INT])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.sym_size.int,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="dim", type=Literal[schema.ScalarDType.INT]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.ScalarDType.INT])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.upsample_nearest2d.vec,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="input", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="output_size", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="scale_factors", type=Literal[schema.ScalarDType.NULL]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.SI8])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.upsample_nearest2d.vec,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="input", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="output_size", type=list[Literal[schema.ScalarDType.INT]]),
                    schema.Argument(name="scale_factors", type=Literal[schema.ScalarDType.NULL]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.view.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="size", type=list[Literal[schema.ScalarDType.INT]]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.SI8])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.aten.view.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="self", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="size", type=list[Literal[schema.ScalarDType.INT]]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.xaten.kv_cache.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="key", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="value", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="layer_idx", type=Literal[schema.ScalarDType.INT]),
                    schema.Argument(name="total_seq_len", type=Literal[schema.ScalarDType.INT]),
                ],
                output=[
                    schema.Argument(name="", type=Literal[schema.TensorDType.SI8]),
                    schema.Argument(name="", type=Literal[schema.TensorDType.SI8]),
                ],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.xaten.kv_cache.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="key", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="value", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="layer_idx", type=Literal[schema.ScalarDType.INT]),
                    schema.Argument(name="total_seq_len", type=Literal[schema.ScalarDType.INT]),
                ],
                output=[
                    schema.Argument(name="", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="", type=Literal[schema.TensorDType.F32]),
                ],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.xaten.rotary_positional_embeddings.default,
            signature=schema.Signature(
                input=[
                    schema.Argument(name="x", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="cos", type=Literal[schema.TensorDType.F32]),
                    schema.Argument(name="sin", type=Literal[schema.TensorDType.F32]),
                ],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.xaten.fan_out.Tensor,
            signature=schema.Signature(
                input=[schema.Argument(name="input", type=Literal[schema.TensorDType.SI8])],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.SI8])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.xaten.fan_out.Tensor,
            signature=schema.Signature(
                input=[schema.Argument(name="input", type=Literal[schema.TensorDType.F32])],
                output=[schema.Argument(name="", type=Literal[schema.TensorDType.F32])],
            ),
            metadata="",
        ),
        schema.Schema(
            engine=_Alpha.ENGINE,
            target=torch.ops.xaten.fan_out.int,
            signature=schema.Signature(
                input=[schema.Argument(name="input", type=Literal[schema.ScalarDType.INT])],
                output=[schema.Argument(name="", type=Literal[schema.ScalarDType.INT])],
            ),
            metadata="",
        ),
    }
)
"""The database of engine operators."""


DATABASE: Set[schema.Schema] = _ENGINE
"""The database of operators of the Alpha hardware platform generation."""
