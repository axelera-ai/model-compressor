# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import collections
import dataclasses
from collections.abc import Set
from typing import cast

import torch
import torch._ops

import model_compressor.ops  # noqa: F401 - register extended ATen operators
import model_compressor.passes.adapt.base as base
import model_compressor.passes.adapt.markers as markers


@dataclasses.dataclass(frozen=True)
class _PortBlock:
    """A block of ports that must share the same semantics.

    Args:
        input: A subset of input ports.
        output: A subset of output ports.

    """

    input: Set[base.nx.Port]
    output: Set[base.nx.Port]


@dataclasses.dataclass(frozen=True)
class _PortIndex:
    """Map ports to the shared-semantics blocks they belong to.

    Args:
        input: A mapping from input ports to the shared-semantics blocks they belong to.
        output: A mapping from output ports to the shared-semantics blocks they belong to.

    """

    input: dict[base.nx.Port, _PortBlock]
    output: dict[base.nx.Port, _PortBlock]


@dataclasses.dataclass(frozen=True)
class EdgeBlock:
    """A block of graph edges incident to a given node that must share the same semantics.

    Args:
        incoming: A subset of the node's incoming edges.
        outgoing: A subset of the node's outgoing edges.

    """

    incoming: set[base.nx.Edge]
    outgoing: set[base.nx.Edge]


_ANY = -1
"""Wildcard for any position in a positional container (i.e., a ``Sequence`` instance)."""


_TARGET_TO_PORT_BLOCKS: dict[torch._ops.OpOverload, list[_PortBlock]] = {
    torch.ops.aten._native_batch_norm_legit_no_training.default: [
        _PortBlock(input=frozenset({("input",)}), output=frozenset()),
        _PortBlock(input=frozenset({("weight",)}), output=frozenset()),
        _PortBlock(input=frozenset({("bias",)}), output=frozenset()),
        _PortBlock(input=frozenset({("running_mean",)}), output=frozenset()),
        _PortBlock(input=frozenset({("running_var",)}), output=frozenset()),
        _PortBlock(input=frozenset({("momentum",)}), output=frozenset()),
        _PortBlock(input=frozenset({("eps",)}), output=frozenset()),
        _PortBlock(input=frozenset(), output=frozenset({("", 0)})),
        _PortBlock(input=frozenset(), output=frozenset({("", 1)})),
        _PortBlock(input=frozenset(), output=frozenset({("", 2)})),
    ],
    torch.ops.aten.add.Tensor: [
        _PortBlock(input=frozenset({("self",)}), output=frozenset()),
        _PortBlock(input=frozenset({("other",)}), output=frozenset()),
        _PortBlock(input=frozenset({("alpha",)}), output=frozenset()),
        _PortBlock(input=frozenset(), output=frozenset({("",)})),
    ],
    torch.ops.aten.cat.default: [
        _PortBlock(input=frozenset({("tensors", _ANY)}), output=frozenset({("",)})),
        _PortBlock(input=frozenset({("dim",)}), output=frozenset()),
    ],
    torch.ops.aten.clone.default: [
        _PortBlock(input=frozenset({("self",)}), output=frozenset({("",)})),
        _PortBlock(input=frozenset({("memory_format",)}), output=frozenset()),
    ],
    torch.ops.aten.constant_pad_nd.default: [
        _PortBlock(input=frozenset({("self",)}), output=frozenset({("",)})),
        _PortBlock(input=frozenset({("pad", _ANY)}), output=frozenset()),
        _PortBlock(input=frozenset({("value",)}), output=frozenset()),
    ],
    torch.ops.aten.convolution.default: [
        _PortBlock(input=frozenset({("input",)}), output=frozenset()),
        _PortBlock(input=frozenset({("weight",)}), output=frozenset()),
        _PortBlock(input=frozenset({("bias",)}), output=frozenset()),
        _PortBlock(input=frozenset({("stride", _ANY)}), output=frozenset()),
        _PortBlock(input=frozenset({("padding", _ANY)}), output=frozenset()),
        _PortBlock(input=frozenset({("dilation", _ANY)}), output=frozenset()),
        _PortBlock(input=frozenset({("transposed",)}), output=frozenset()),
        _PortBlock(input=frozenset({("output_padding", _ANY)}), output=frozenset()),
        _PortBlock(input=frozenset({("groups",)}), output=frozenset()),
        _PortBlock(input=frozenset(), output=frozenset({("",)})),
    ],
    torch.ops.aten.embedding.default: [
        _PortBlock(input=frozenset({("weight",)}), output=frozenset({("",)})),
        _PortBlock(input=frozenset({("indices",)}), output=frozenset()),
        _PortBlock(input=frozenset({("padding_idx",)}), output=frozenset()),
        _PortBlock(input=frozenset({("scale_grad_by_freq",)}), output=frozenset()),
        _PortBlock(input=frozenset({("sparse",)}), output=frozenset()),
    ],
    torch.ops.aten.expand.default: [
        _PortBlock(input=frozenset({("self",)}), output=frozenset({("",)})),
        _PortBlock(input=frozenset({("size", _ANY)}), output=frozenset()),
        _PortBlock(input=frozenset({("implicit",)}), output=frozenset()),
    ],
    torch.ops.aten.gelu.default: [
        _PortBlock(input=frozenset({("self",)}), output=frozenset()),
        _PortBlock(input=frozenset({("approximate",)}), output=frozenset()),
        _PortBlock(input=frozenset(), output=frozenset({("",)})),
    ],
    torch.ops.aten.layer_norm.default: [
        _PortBlock(input=frozenset({("input",)}), output=frozenset()),
        _PortBlock(input=frozenset({("normalized_shape", _ANY)}), output=frozenset()),
        _PortBlock(input=frozenset({("weight",)}), output=frozenset()),
        _PortBlock(input=frozenset({("bias",)}), output=frozenset()),
        _PortBlock(input=frozenset({("eps",)}), output=frozenset()),
        _PortBlock(input=frozenset({("cudnn_enable",)}), output=frozenset()),
        _PortBlock(input=frozenset(), output=frozenset({("",)})),
    ],
    torch.ops.aten.linear.default: [
        _PortBlock(input=frozenset({("input",)}), output=frozenset()),
        _PortBlock(input=frozenset({("weight",)}), output=frozenset()),
        _PortBlock(input=frozenset({("bias",)}), output=frozenset()),
        _PortBlock(input=frozenset(), output=frozenset({("",)})),
    ],
    torch.ops.aten.max_pool2d_with_indices.default: [
        _PortBlock(input=frozenset({("self",)}), output=frozenset({("", 0)})),
        _PortBlock(input=frozenset({("kernel_size", _ANY)}), output=frozenset()),
        _PortBlock(input=frozenset({("stride", _ANY)}), output=frozenset()),
        _PortBlock(input=frozenset({("padding", _ANY)}), output=frozenset()),
        _PortBlock(input=frozenset({("dilation", _ANY)}), output=frozenset()),
        _PortBlock(input=frozenset({("ceil_mode",)}), output=frozenset()),
        _PortBlock(input=frozenset(), output=frozenset({("", 1)})),
    ],
    torch.ops.aten.mean.dim: [
        _PortBlock(input=frozenset({("self",)}), output=frozenset()),
        _PortBlock(input=frozenset({("dim", _ANY)}), output=frozenset()),
        _PortBlock(input=frozenset({("keepdim",)}), output=frozenset()),
        _PortBlock(input=frozenset({("dtype",)}), output=frozenset()),
        _PortBlock(input=frozenset(), output=frozenset({("",)})),
    ],
    torch.ops.aten.mul.Tensor: [
        _PortBlock(input=frozenset({("self",)}), output=frozenset()),
        _PortBlock(input=frozenset({("other",)}), output=frozenset()),
        _PortBlock(input=frozenset(), output=frozenset({("",)})),
    ],
    torch.ops.aten.permute.default: [
        _PortBlock(input=frozenset({("self",)}), output=frozenset({("",)})),
        _PortBlock(input=frozenset({("dims", _ANY)}), output=frozenset()),
    ],
    torch.ops.aten.relu.default: [
        _PortBlock(input=frozenset({("self",)}), output=frozenset()),
        _PortBlock(input=frozenset(), output=frozenset({("",)})),
    ],
    torch.ops.aten.rms_norm.default: [
        _PortBlock(input=frozenset({("input",)}), output=frozenset()),
        _PortBlock(input=frozenset({("normalized_shape", _ANY)}), output=frozenset()),
        _PortBlock(input=frozenset({("weight",)}), output=frozenset()),
        _PortBlock(input=frozenset({("eps",)}), output=frozenset()),
        _PortBlock(input=frozenset(), output=frozenset({("",)})),
    ],
    torch.ops.aten.scaled_dot_product_attention.default: [
        _PortBlock(input=frozenset({("query",)}), output=frozenset()),
        _PortBlock(input=frozenset({("key",)}), output=frozenset()),
        _PortBlock(input=frozenset({("value",)}), output=frozenset()),
        _PortBlock(input=frozenset({("attn_mask",)}), output=frozenset()),
        _PortBlock(input=frozenset({("dropout_p",)}), output=frozenset()),
        _PortBlock(input=frozenset({("is_causal",)}), output=frozenset()),
        _PortBlock(input=frozenset({("scale",)}), output=frozenset()),
        _PortBlock(input=frozenset({("enable_gqa",)}), output=frozenset()),
        _PortBlock(input=frozenset(), output=frozenset({("",)})),
    ],
    torch.ops.aten.silu.default: [
        _PortBlock(input=frozenset({("self",)}), output=frozenset()),
        _PortBlock(input=frozenset(), output=frozenset({("",)})),
    ],
    torch.ops.aten.slice.Tensor: [
        _PortBlock(input=frozenset({("self",)}), output=frozenset({("",)})),
        _PortBlock(input=frozenset({("dim",)}), output=frozenset()),
        _PortBlock(input=frozenset({("start",)}), output=frozenset()),
        _PortBlock(input=frozenset({("end",)}), output=frozenset()),
        _PortBlock(input=frozenset({("step",)}), output=frozenset()),
    ],
    torch.ops.aten.split_with_sizes.default: [
        _PortBlock(input=frozenset({("self",)}), output=frozenset({("", _ANY)})),
        _PortBlock(input=frozenset({("split_sizes", _ANY)}), output=frozenset()),
        _PortBlock(input=frozenset({("dim",)}), output=frozenset()),
    ],
    torch.ops.aten.sym_size.int: [
        _PortBlock(input=frozenset({("self",)}), output=frozenset()),
        _PortBlock(input=frozenset({("dim",)}), output=frozenset()),
        _PortBlock(input=frozenset(), output=frozenset({("",)})),
    ],
    torch.ops.aten.upsample_nearest2d.vec: [
        _PortBlock(input=frozenset({("input",)}), output=frozenset({("",)})),
        _PortBlock(input=frozenset({("output_size", _ANY)}), output=frozenset()),
        _PortBlock(input=frozenset({("scale_factors", _ANY)}), output=frozenset()),
    ],
    torch.ops.aten.view.default: [
        _PortBlock(input=frozenset({("self",)}), output=frozenset({("",)})),
        _PortBlock(input=frozenset({("size", _ANY)}), output=frozenset()),
    ],
    torch.ops.xaten.kv_cache.default: [
        _PortBlock(input=frozenset({("key",)}), output=frozenset({("", 0)})),
        _PortBlock(input=frozenset({("value",)}), output=frozenset({("", 1)})),
        _PortBlock(input=frozenset({("layer_idx",)}), output=frozenset()),
        _PortBlock(input=frozenset({("total_seq_len",)}), output=frozenset()),
    ],
    torch.ops.xaten.rotary_positional_embeddings.default: [
        _PortBlock(input=frozenset({("x",)}), output=frozenset()),
        _PortBlock(input=frozenset({("cos",)}), output=frozenset()),
        _PortBlock(input=frozenset({("sin",)}), output=frozenset()),
        _PortBlock(input=frozenset(), output=frozenset({("",)})),
    ],
    torch.ops.xaten.fan_out.Tensor: [
        _PortBlock(input=frozenset({("input",)}), output=frozenset({("",)})),
    ],
    torch.ops.xaten.fan_out.int: [
        _PortBlock(input=frozenset({("input",)}), output=frozenset({("",)})),
    ],
}
"""For each operator, partition its ports into blocks that must share the same semantics."""


assert (frozenset(_TARGET_TO_PORT_BLOCKS) & markers.MARKERS_CODEC) == frozenset()


def _index_port_blocks(
    target_to_port_blocks: dict[torch._ops.OpOverload, list[_PortBlock]],
) -> dict[torch._ops.OpOverload, _PortIndex]:
    """Convert collections of shared-semantics port blocks to mappings from ports to port blocks.

    Args:
        target_to_port_blocks: A mapping from operators to shared-semantics port blocks.

    Returns:
        A mapping from operators to mappings from ports to shared-semantics port blocks.

    """
    target_to_port_index = {}
    for target, port_blocks in target_to_port_blocks.items():
        port_index = _PortIndex(input={}, output={})
        for block in port_blocks:
            port_index.input.update(dict.fromkeys(block.input, block))
            port_index.output.update(dict.fromkeys(block.output, block))
        target_to_port_index[target] = port_index
    return target_to_port_index


_TARGET_TO_PORT_INDEX = _index_port_blocks(_TARGET_TO_PORT_BLOCKS)
"""For each operator, map ports to the corresponding shared-semantics port blocks."""


def compute_edge_blocks(graph: base.nx.Graph, node: base.nx.Node) -> list[EdgeBlock]:
    """Partition the edges incident to a node according to the ports they feed/are fed from.

    Args:
        graph: The graph to which the node whose incident edges should be partitioned belongs.
        node: The node whose incident edges should be partitioned.

    Raises:
        ValueError: The node points to an operator whose commuting port blocks are not defined.

    """
    target = base.nx.get_annotation(graph, node).target
    if target not in _TARGET_TO_PORT_INDEX:
        raise ValueError(
            f"Node {node} must point to an operator whose port blocks are defined, but has target {target}"
        )
    port_index = _TARGET_TO_PORT_INDEX[cast(torch._ops.OpOverload, target)]
    # partition edges according to the ports they feed into/are fed from
    port_block_to_edge_block: dict[_PortBlock, EdgeBlock] = collections.defaultdict(
        lambda: EdgeBlock(incoming=set(), outgoing=set())
    )
    for edge in graph.in_edges(node, keys=True):
        (_, _, (_, input_port)) = edge
        head, *path = input_port
        if len(path) == 0:
            look_up_port = input_port
        else:
            look_up_port = input_port if (input_port in port_index.input) else (head, _ANY)
        port_block = port_index.input[look_up_port]
        edge_block = port_block_to_edge_block[port_block]
        edge_block.incoming.add(edge)
    for edge in graph.out_edges(node, keys=True):
        (_, _, (output_port, _)) = edge
        head, *path = output_port
        if len(path) == 0:
            look_up_port = output_port
        else:
            (idx,) = path
            head = ""  # patch ``"getitem"``
            look_up_port = (head, idx) if ((head, idx) in port_index.output) else (head, _ANY)
        port_block = port_index.output[look_up_port]
        edge_block = port_block_to_edge_block[port_block]
        edge_block.outgoing.add(edge)
    return list(port_block_to_edge_block.values())
