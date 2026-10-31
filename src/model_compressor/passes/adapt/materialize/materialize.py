# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import collections
import enum
from collections.abc import Callable
from typing import Literal, cast

import torch
import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base

import model_compressor.ops  # noqa: F401 - register extended ATen (XATen) operators
import model_compressor.passes.adapt.base as base
import model_compressor.passes.adapt.markers as markers
import model_compressor.passes.annotate.base as annotate
import model_compressor.passes.utils.state as _state


class _Compression(enum.Enum):
    CODEC = enum.auto()
    DECODE = enum.auto()
    ENCODE = enum.auto()
    RECODE = enum.auto()
    NULL = enum.auto()


_COMPRESSION_TO_TARGET = {
    _Compression.CODEC: torch.ops.xaten.codec.default,
    _Compression.DECODE: torch.ops.xaten.decode.default,
    _Compression.ENCODE: torch.ops.xaten.encode.default,
    _Compression.RECODE: torch.ops.xaten.recode.default,
    _Compression.NULL: torch.ops.xaten.codec.default,
}


def _is_lossless(graph: base.nx.Graph, node: base.nx.Node) -> bool:
    annotation = base.nx.get_annotation(graph, node)
    assert annotation.target in markers.MARKERS_CODEC
    if annotation.target in markers.MARKERS_INPUT_CODEC:
        semantics = base.nx.get_output_port_semantics(graph, node, ("",))
    elif annotation.target in markers.MARKERS_OUTPUT_CODEC:
        semantics = base.nx.get_input_port_semantics(graph, node, ("input",))
    else:
        raise ValueError(f"Node must be a codec marker, is {annotation.target}")
    return semantics.is_true


def _materialize(state: _state.State, graph: base.nx.Graph) -> base.nx.PassResult:
    semantics_to_subgraph = {}
    for edge in graph.edges(keys=True):
        subgraph = base.nx.get_subgraph(graph, edge)
        semantics_to_subgraph[subgraph.semantics] = subgraph
    lossless: set[base.nx.Node] = set()
    lossy: set[base.nx.Node] = set()
    for node in graph.nodes:
        annotation = base.nx.get_annotation(graph, node)
        if annotation.target in markers.MARKERS_CODEC:
            set_ = lossless if _is_lossless(graph, node) else lossy
            set_.add(node)
    compression_to_matches = collections.defaultdict(set)
    for subgraph in semantics_to_subgraph.values():
        if subgraph.semantics.is_fake:
            nodes = {
                node for node in subgraph.nodes if (base.nx.get_annotation(graph, node).target in markers.MARKERS_CODEC)
            }
            decoders = {node for node in subgraph.exits if (node in lossy)}
            lossless_exits = {node for node in subgraph.exits if (node in lossless)}
            decoders |= {pred for node in lossless_exits for pred in graph.predecessors(node) if (pred in lossy)}
            encoders = {node for node in subgraph.entries if (node in lossy)}
            null = nodes - (decoders | encoders)
            compression_to_matches[_Compression.DECODE].update(decoders)
            compression_to_matches[_Compression.ENCODE].update(encoders)
            compression_to_matches[_Compression.NULL].update(null)
    decoders = compression_to_matches[_Compression.DECODE]
    encoders = compression_to_matches[_Compression.ENCODE]
    recoders = decoders & encoders
    decoders -= recoders
    encoders -= recoders
    compression_to_matches[_Compression.DECODE] = decoders
    compression_to_matches[_Compression.ENCODE] = encoders
    compression_to_matches[_Compression.RECODE] = recoders
    if len(lossless | lossy) == 0:
        return base.nx.PassResult(graph, False)
    for compression, matches in compression_to_matches.items():
        if compression in (_Compression.CODEC, _Compression.NULL):
            continue
        target = _COMPRESSION_TO_TARGET[compression]
        for match in matches:
            old_node = match
            _ = _replace(state, graph, old_node, target)
    return base.nx.PassResult(graph, True)


def _replace(
    state: _state.State, graph: base.nx.Graph, old_node: base.nx.Node, target: torch._ops.OpOverload
) -> base.nx.Node:
    old_incoming = list(graph.in_edges(old_node, keys=True))
    old_outgoing = list(graph.out_edges(old_node, keys=True))
    new_incoming, new_outgoing = [], []
    new_node = state.book(target)
    graph.add_node(new_node)
    old_annotation = base.nx.get_annotation(graph, old_node)
    in_semantics = base.nx.get_input_port_semantics(graph, old_node, ("input",))
    out_semantics = base.nx.get_output_port_semantics(graph, old_node, ("",))
    in_type = out_semantics.latent if (target in (torch.ops.xaten.recode.default,)) else in_semantics.latent
    out_type = out_semantics.latent
    new_annotation = base.nx.Annotation(
        op="call_function",
        target=target,
        kwargs=old_annotation.kwargs,
        value=old_annotation.value,
        annotation=annotate.Annotation(
            source=annotate.Schema(
                engine=old_annotation.source.engine,
                target=target,
                signature=annotate.Signature(
                    input=[annotate.Argument(name="input", type=Literal[in_semantics.manifest])],
                    output=[annotate.Argument(name="", type=Literal[out_semantics.manifest])],
                ),
                metadata="",
            ),
            chosen=annotate.Schema(
                engine=old_annotation.chosen.engine,
                target=target,
                signature=annotate.Signature(
                    input=[annotate.Argument(name="input", type=Literal[in_type])],
                    output=[annotate.Argument(name="", type=Literal[out_type])],
                ),
                metadata="",
            ),
        ),
    )
    base.nx.set_annotation(graph, new_node, new_annotation)
    for old_in_edge in old_incoming:
        (pred, _, (output_port_pred, _)) = old_in_edge
        new_in_edge = (pred, new_node, (output_port_pred, ("input",)))
        new_incoming.append(new_in_edge)
        graph.add_edge(pred, new_node, key=(output_port_pred, ("input",)))
        base.nx.set_semantics(graph, new_in_edge, base.nx.get_semantics(graph, old_in_edge))
        base.nx.set_subgraph(graph, new_in_edge, base.nx.get_subgraph(graph, old_in_edge))
    for old_out_edge in old_outgoing:
        (_, succ, (_, input_port_succ)) = old_out_edge
        new_out_edge = (new_node, succ, (("",), input_port_succ))
        new_outgoing.append(new_out_edge)
        graph.add_edge(new_node, succ, key=(("",), input_port_succ))
        base.nx.set_semantics(graph, new_out_edge, base.nx.get_semantics(graph, old_out_edge))
        base.nx.set_subgraph(graph, new_out_edge, base.nx.get_subgraph(graph, old_out_edge))
    for pred, _, (output_port_pred, _) in old_incoming:
        graph.remove_edge(pred, old_node, key=(output_port_pred, ("input",)))
    for _, succ, (_, input_port_succ) in old_outgoing:
        graph.remove_edge(old_node, succ, key=(("",), input_port_succ))
    graph.remove_node(old_node)
    # update the subgraph
    upstream_subgraph, *_ = [base.nx.get_subgraph(graph, edge) for edge in new_incoming]
    downstream_subgraph, *_ = [base.nx.get_subgraph(graph, edge) for edge in new_outgoing]
    if downstream_subgraph is not upstream_subgraph:
        upstream_subgraph.nodes.add(new_node)
        upstream_subgraph.nodes.remove(old_node)
        upstream_subgraph.edges.update(new_incoming)
        upstream_subgraph.edges.difference_update(old_incoming)
        for old_edge, new_edge in zip(old_incoming, new_incoming, strict=False):
            if old_edge in upstream_subgraph.entrying:
                upstream_subgraph.entrying.add(new_edge)
                upstream_subgraph.entrying.remove(old_edge)
            if old_edge in upstream_subgraph.exiting:
                upstream_subgraph.exiting.add(new_edge)
                upstream_subgraph.exiting.remove(old_edge)
        if old_node in upstream_subgraph.dominators:
            upstream_subgraph.dominators.add(new_node)
            upstream_subgraph.dominators.remove(old_node)
        downstream_subgraph.nodes.add(new_node)
        downstream_subgraph.nodes.remove(old_node)
        downstream_subgraph.edges.update(new_outgoing)
        downstream_subgraph.edges.difference_update(old_outgoing)
        for old_edge, new_edge in zip(old_outgoing, new_outgoing, strict=False):
            if old_edge in downstream_subgraph.entrying:
                downstream_subgraph.entrying.add(new_edge)
                downstream_subgraph.entrying.remove(old_edge)
            if old_edge in downstream_subgraph.exiting:
                downstream_subgraph.exiting.add(new_edge)
                downstream_subgraph.exiting.remove(old_edge)
        if old_node in downstream_subgraph.dominators:
            downstream_subgraph.dominators.add(new_node)
            downstream_subgraph.dominators.remove(old_node)
    else:
        subgraph = upstream_subgraph
        subgraph.nodes.add(new_node)
        subgraph.nodes.remove(old_node)
        subgraph.edges.update(new_incoming)
        subgraph.edges.update(new_outgoing)
        subgraph.edges.difference_update(old_incoming)
        subgraph.edges.difference_update(old_outgoing)
        for old_edge, new_edge in zip(old_incoming + old_outgoing, new_incoming + new_outgoing, strict=False):
            if old_edge in subgraph.entrying:
                subgraph.entrying.add(new_edge)
                subgraph.entrying.remove(old_edge)
            if old_edge in subgraph.exiting:
                subgraph.exiting.add(new_edge)
                subgraph.exiting.remove(old_edge)
        if old_node in subgraph.dominators:
            subgraph.dominators.add(new_node)
            subgraph.dominators.remove(old_node)
    state.free(cast(Callable, old_annotation.target), cast(str, old_node))
    return new_node


class Materialize(pass_base.PassBase):
    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        state = _state.State(graph_module.graph)
        graph = base.convert_fx_to_nx(graph_module)
        graph = base.convert_fx_subgraphs_to_nx_subgraphs(graph_module, graph)
        pass_result = _materialize(state, graph)
        graph_module = base.convert_nx_to_fx(graph)
        graph_module = base.convert_nx_subgraphs_to_fx_subgraphs(graph, graph_module)
        return pass_base.PassResult(graph_module, pass_result.modified)
