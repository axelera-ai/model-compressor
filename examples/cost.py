# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from __future__ import annotations

import builtins
import collections
import dataclasses
import typing
from collections.abc import Set
from typing import cast, get_args, get_origin

import model_compressor.database.schema as schema
import model_compressor.passes.adapt.markers as markers
import model_compressor.passes.annotate.base as annotate
import model_compressor.passes.utils.evaluate as evaluate
import model_compressor.passes.utils.rgetattr as rgetattr
import torch
import torch.fx as fx

_TENSOR_DTYPE_TO_BYTES: dict[schema.TensorDType, int] = {
    schema.TensorDType.SI8: 1,
    schema.TensorDType.SI32: 4,
    schema.TensorDType.SI64: 8,
    schema.TensorDType.F16: 2,
    schema.TensorDType.F32: 4,
}
"""Map data types to the storage required by their instances."""


@dataclasses.dataclass
class _Parameter:
    """A model parameter.

    Args:
        node: The node pointing to the parameter.
        type: The parameter's data type.
        shape: The parameter's shape.

    """

    node: fx.Node
    type: schema.TensorDType
    shape: torch.Size

    @property
    def name(self) -> str:
        """The parameter's name."""
        return self.node.name

    @property
    def bytes(self) -> int:
        """The parameter's storage."""
        return _TENSOR_DTYPE_TO_BYTES[self.type] * self.shape.numel()


@dataclasses.dataclass
class _Read:
    """A read performed by an operator.

    Args:
        node: The node pointing to the operator performing the read.
        name: The name of the parameter corresponding to the read argument.
        type: The read argument's data type.
        shape: The read argument's shape.

    """

    node: fx.Node
    name: str
    type: schema.TensorDType
    shape: torch.Size

    @property
    def target(self) -> torch._ops.OpOverload:
        """The operator performing the read."""
        return cast(torch._ops.OpOverload, self.node.target)

    @property
    def bytes(self) -> int:
        """The read argument's storage."""
        return _TENSOR_DTYPE_TO_BYTES[self.type] * self.shape.numel()


@dataclasses.dataclass
class _Codec:
    """An operator performing compression/decompression/recompression.

    Args:
        node: The node pointing to the compression/decompression/recompression operator.
        type: The operator input's data type.
        shape: The operator input's shape.

    """

    node: fx.Node
    type: schema.TensorDType
    shape: torch.Size

    @property
    def name(self) -> str:
        """The operator's name."""
        return self.node.name

    @property
    def numel(self) -> int:
        """The operator input's number of items."""
        return self.shape.numel()


@dataclasses.dataclass
class _Traffic:
    """Summarize a model's traffic - as approximated by reads.

    Args:
        reads: The reads performed by the model's operators.

    """

    reads: list[_Read] = dataclasses.field(default_factory=list)

    @property
    def bytes(self) -> int:
        return sum(read.bytes for read in self.reads)


@dataclasses.dataclass
class _Codecs:
    """Summarize the encoding/decoding/recoding operators in a model.

    Args:
        ops: The model's codec operators.

    """

    ops: list[_Codec] = dataclasses.field(default_factory=list)

    @property
    def count(self) -> int:
        return sum(op.numel for op in self.ops)


_NOOP = (*markers.MARKERS_CODEC, torch.ops.xaten.fan_out.Tensor, torch.ops.xaten.fan_out.int)


def analyze_model_cost(
    graph_module: fx.GraphModule, input: dict[str, torch.Tensor]
) -> tuple[_Storage, _Traffic, _Codecs]:
    """Analyze a model's storage, traffic, and encoding/decoding/recoding operators.

    Args:
        graph_module: A ``GraphModule``.
        input: An input for the ``GraphModule``.

    Returns:
        A triple whose items have the following semantics:
            * a summary of the model's storage state;
            * a summary of the model's data traffic - as approximated by reads;
            * a summary of the model's encoding/decoding/recoding operators.

    """
    storage = _analyze_storage(graph_module)
    node_to_execution_context = evaluate.evaluate_execution_contexts(graph_module, input)
    traffic = _Traffic()
    for node in graph_module.graph.nodes:
        if (node.op == "call_function") and isinstance(node.target, torch._ops.OpOverload):
            if node.target not in _NOOP:
                annotation = annotate.get_annotation(node)
                execution_context = node_to_execution_context[node]
                for parameter, argument in zip(annotation.chosen.signature.input, execution_context.input, strict=True):
                    assert parameter.name == argument.name
                    match get_origin(parameter.type):
                        case builtins.list:
                            type, *_ = get_args(parameter.type)
                            type, *_ = get_args(type)
                            match type:
                                case schema.TensorDType():
                                    for idx, item in enumerate(argument.value):
                                        read = _Read(node, f"{parameter.name}.{idx}", type, item.shape)
                                        traffic.reads.append(read)
                                case schema.ScalarDType():
                                    pass
                                case _:
                                    raise ValueError
                        case typing.Literal:
                            type, *_ = get_args(parameter.type)
                            match type:
                                case schema.TensorDType():
                                    read = _Read(node, parameter.name, type, argument.value.shape)
                                    traffic.reads.append(read)
                                case schema.ScalarDType():
                                    pass
                                case _:
                                    raise ValueError
                        case _:
                            raise ValueError
    codecs = _Codecs()
    for read in traffic.reads:
        if read.target in (
            torch.ops.xaten.decode.default,
            torch.ops.xaten.encode.default,
            torch.ops.xaten.recode.default,
        ):
            codec = _Codec(read.node, read.type, read.shape)
            codecs.ops.append(codec)

    param_nodes = {param.node: param for param in storage.parameters}
    codec_nodes = {op.node: op for op in codecs.ops}
    reads_nodes = collections.defaultdict(list)
    for read in traffic.reads:
        reads_nodes[read.node].append(read)
    for node in graph_module.graph.nodes:
        if node.target is torch.ops.xaten.encode.default:
            arg = node.kwargs["input"]
            if (arg.op == "get_attr") and (arg in param_nodes):
                codecs.ops.remove(codec_nodes[node])
                for read in reads_nodes[node]:
                    traffic.reads.remove(read)
    return storage, traffic, codecs


@dataclasses.dataclass
class _Storage:
    """Summarize a model's storage state.

    Args:
        foldable: The lossy codec markers that can be folded at compilation time.
        parameters: The model parameters.

    """

    foldable: set[fx.Node] = dataclasses.field(default_factory=set)
    parameters: list[_Parameter] = dataclasses.field(default_factory=list)

    @property
    def bytes(self) -> int:
        """The total storage required by the model."""
        return sum(parameter.bytes for parameter in self.parameters)


def _analyze_storage(graph_module: fx.GraphModule) -> _Storage:
    """Analyze a model's storage state.

    Args:
        graph_module: A ``GraphModule``.

    Returns:
       The models' storage state.

    """
    # Assumption:
    #   * in the source ``GraphModule``, each ``"get_attr"`` node is consumed by one or more
    #     ``"call_function"`` nodes;
    #   * during adaptation, each ``"call_function"`` node is prepended ``mark_input_codec``
    #     nodes;
    #   * after adaptation, each ``"get_attr"`` node is consumed by one or more
    #     ``mark_input_codec`` nodes.
    node_to_users: dict[fx.Node, Set[fx.Node]] = {}
    for node in graph_module.graph.nodes:
        if node.op == "get_attr":
            users = frozenset(node.users)
            are_all_mark_input_codecs = all(
                (user.target in (*markers.MARKERS_INPUT_CODEC, torch.ops.xaten.encode.default)) for user in users
            )
            assert are_all_mark_input_codecs
            node_to_users[node] = users
    storage = _Storage()
    for node, users in node_to_users.items():
        value = rgetattr.rgetattr(graph_module, node.target)
        lossless = frozenset(user for user in users if (user.target in markers.MARKERS_INPUT_CODEC))
        lossy = users - lossless
        if len(lossless) > 0:
            user, *_ = lossless
            type = _get_codec_output_type(user)
            parameter = _Parameter(node, type, value.shape)
            storage.parameters.append(parameter)
        if len(lossy) > 0:
            storage.foldable |= lossy
            user, *_ = lossy
            type = _get_codec_output_type(user)
            parameter = _Parameter(node, type, value.shape)
            storage.parameters.append(parameter)
    return storage


def _get_codec_output_type(node: fx.Node) -> schema.DType:
    """Get a codec's output data type."""
    annotation = annotate.get_annotation(node)
    chosen, *_ = annotation.chosen.signature.output
    type, *_ = get_args(chosen.type)
    return type
