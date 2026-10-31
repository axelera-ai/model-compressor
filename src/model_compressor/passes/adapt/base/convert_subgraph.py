# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import collections
import copy
import operator
from typing import Any, cast

import torch
import torch.fx as fx

import model_compressor.passes.adapt.base as base
import model_compressor.passes.adapt.base.convert as convert


def _convert_fx_node_to_kwargs_to_stubbed_and_edges(
    fx_node_to_node: dict[fx.Node, base.nx.Node],
    fx_node_to_kwargs: dict[fx.Node, torch.fx.immutable_collections.immutable_dict],
) -> tuple[dict[str, Any], list[base.nx.Edge]]:
    (fx_node, kwargs), *_ = fx_node_to_kwargs.items()
    stubbed, patch = convert.unzip_kwargs(kwargs)
    node = fx_node_to_node[fx_node]
    edges = []
    output_port_pred: base.nx.Port
    for input_port, fx_pred in patch.items():
        if (fx_pred.op == "call_function") and (fx_pred.target is operator.getitem):
            getitem = fx_pred
            fx_pred, idx = cast(tuple[fx.Node, int], getitem.args)
            output_port_pred = (getitem.name, idx)
        else:
            output_port_pred = ("",)
        pred = fx_node_to_node[fx_pred]
        edge = (pred, node, (output_port_pred, input_port))
        edges.append(edge)
    return stubbed, edges


def convert_fx_subgraphs_to_nx_subgraphs(fx_graph_module: fx.GraphModule, graph: base.nx.Graph) -> base.nx.Graph:
    fx_node_to_node = {
        fx_node: fx_node.name for fx_node in fx_graph_module.graph.nodes if (fx_node.name in graph.nodes)
    }
    semantics_to_fx_subgraph = {}
    for fx_node in fx_graph_module.graph.nodes:
        if base.fx.has_kwarg_to_subgraph(fx_node):
            for fx_subgraph in base.fx.get_kwarg_to_subgraph(fx_node).values():
                semantics_to_fx_subgraph[fx_subgraph.semantics] = fx_subgraph
    # convert FX subgraphs to NetworkX subgraphs
    semantics_to_subgraph = {}
    for semantics, fx_subgraph in semantics_to_fx_subgraph.items():
        subgraph = base.nx.Subgraph(semantics=semantics)
        subgraph.nodes |= {fx_node_to_node[fx_node] for fx_node in fx_subgraph.nodes}
        subgraph.dominators |= {fx_node_to_node[fx_node] for fx_node in fx_subgraph.dominators}
        for fx_node, kwargs in fx_subgraph.edges.items():
            _, edges = _convert_fx_node_to_kwargs_to_stubbed_and_edges(fx_node_to_node, {fx_node: kwargs})
            subgraph.edges |= set(edges)
        for fx_node, kwargs in fx_subgraph.entrying.items():
            _, edges = _convert_fx_node_to_kwargs_to_stubbed_and_edges(fx_node_to_node, {fx_node: kwargs})
            subgraph.entrying |= set(edges)
        for fx_node, kwargs in fx_subgraph.exiting.items():
            _, edges = _convert_fx_node_to_kwargs_to_stubbed_and_edges(fx_node_to_node, {fx_node: kwargs})
            subgraph.exiting |= set(edges)
        semantics_to_subgraph[semantics] = subgraph
    # annotate the ``base.nx.Graph``
    for semantics, subgraph in semantics_to_subgraph.items():
        for edge in subgraph.edges:
            base.nx.set_semantics(graph, edge, semantics)
            base.nx.set_subgraph(graph, edge, subgraph)
    return graph


def _convert_stubbed_and_edges_to_fx_node_to_kwargs(
    node_to_fx_node: dict[base.nx.Node, fx.Node],
    stubbed: dict[str, Any],
    edges: list[base.nx.Edge],
) -> dict[fx.Node, torch.fx.immutable_collections.immutable_dict]:
    kwargs: dict[str, Any] = {}
    for edge in edges:
        pred, node, (output_port_pred, input_port) = edge
        # select the patch
        output_port_pred_head, *output_port_pred_path = output_port_pred
        pred = pred if (len(output_port_pred_path) == 0) else output_port_pred_head
        fx_pred = node_to_fx_node[pred]
        # select the kwarg to patch
        input_port_head, *input_port_path = input_port
        kwarg = kwargs.get(input_port_head, copy.deepcopy(stubbed[input_port_head]))
        kwargs[input_port_head] = kwarg
        if len(input_port_path) == 0:
            kwargs[input_port_head] = fx_pred
        else:
            (idx,) = input_port_path
            kwargs[input_port_head][idx] = fx_pred
    for key, value in kwargs.items():
        if isinstance(value, list):
            kwargs[key] = torch.fx.immutable_collections.immutable_list(value)
    fx_node = node_to_fx_node[node]
    return {fx_node: torch.fx.immutable_collections.immutable_dict(kwargs)}


def _convert_frontier_edges_to_fx_node_to_kwargs(
    graph: base.nx.Graph,
    node_to_fx_node: dict[base.nx.Node, fx.Node],
    edges: set[base.nx.Edge],
) -> dict[fx.Node, torch.fx.immutable_collections.immutable_dict]:
    """Build consumer-keyed kwargs for a *frontier* edge set, patching only those edges.

    A frontier set (``entrying``/``exiting``/``dominating``) may hold only a subset of a
    consumer's same-semantics input edges - e.g. a multi-input ``cat`` whose skip input is an
    entry while its sibling input is interior to the subgraph. Building the kwargs from only the
    frontier edges leaves the sibling ports as ``_Stub`` sentinels (the stubbed template comes
    from ``annotation.kwargs``), so the inverse conversion reconstructs exactly ``edges`` and does
    not resurrect the interior siblings.
    """
    node_to_edges: dict[base.nx.Node, list[base.nx.Edge]] = collections.defaultdict(list)
    for edge in edges:
        (_, node, _) = edge
        node_to_edges[node].append(edge)
    fx_node_to_kwargs: dict[fx.Node, torch.fx.immutable_collections.immutable_dict] = {}
    for node, node_edges in node_to_edges.items():
        annotation = base.nx.get_annotation(graph, node)
        fx_node_to_kwargs.update(
            _convert_stubbed_and_edges_to_fx_node_to_kwargs(node_to_fx_node, annotation.kwargs, node_edges)
        )
    return fx_node_to_kwargs


def convert_nx_subgraphs_to_fx_subgraphs(graph: base.nx.Graph, fx_graph_module: fx.GraphModule) -> fx.GraphModule:
    node_to_fx_node = {node.name: node for node in fx_graph_module.graph.nodes}
    semantics_to_subgraph = {}
    for edge in graph.edges:
        subgraph = base.nx.get_subgraph(graph, edge)
        semantics_to_subgraph[subgraph.semantics] = subgraph
    semantics_to_node_to_fx_node: dict[base.nx.Semantics, dict[base.nx.Node, fx.Node]] = collections.defaultdict(dict)
    semantics_to_edge_to_fx_node_to_kwargs: dict[
        base.nx.Semantics, dict[base.nx.Edge, dict[fx.Node, torch.fx.immutable_collections.immutable_dict]]
    ] = collections.defaultdict(dict)
    semantics_to_node_to_edges: dict[base.nx.Semantics, dict[base.nx.Node, list[base.nx.Edge]]] = (
        collections.defaultdict(lambda: collections.defaultdict(list))
    )
    for node in graph.nodes:
        for edge in graph.in_edges(node, keys=True):
            semantics = base.nx.get_semantics(graph, edge)
            semantics_to_node_to_edges[semantics][node].append(edge)
    for semantics, node_to_edges in semantics_to_node_to_edges.items():
        for node, edges in node_to_edges.items():
            annotation = base.nx.get_annotation(graph, node)
            fx_node_to_kwargs = _convert_stubbed_and_edges_to_fx_node_to_kwargs(
                node_to_fx_node, annotation.kwargs, edges
            )
            (fx_node, _), *_ = fx_node_to_kwargs.items()
            semantics_to_node_to_fx_node[semantics][node] = fx_node
            for edge in edges:
                semantics_to_edge_to_fx_node_to_kwargs[semantics][edge] = fx_node_to_kwargs
    # convert NetworkX subgraphs to FX subgraphs
    semantics_to_fx_subgraph = {}
    for semantics, subgraph in semantics_to_subgraph.items():
        fx_subgraph = base.fx.Subgraph(
            semantics=semantics,
            nodes={node_to_fx_node[node] for node in subgraph.nodes},
            edges={},
            entries={node_to_fx_node[node] for node in subgraph.entries},
            exits={node_to_fx_node[node] for node in subgraph.exits},
            dominators={node_to_fx_node[node] for node in subgraph.dominators},
            entrying={},
            exiting={},
            dominating={},
        )
        edge_to_fx_node_to_kwargs = semantics_to_edge_to_fx_node_to_kwargs[semantics]
        for edge in subgraph.edges:
            fx_node_to_kwargs = edge_to_fx_node_to_kwargs[edge]
            fx_subgraph.edges.update(fx_node_to_kwargs)
        fx_subgraph.entrying.update(
            _convert_frontier_edges_to_fx_node_to_kwargs(graph, node_to_fx_node, subgraph.entrying)
        )
        fx_subgraph.exiting.update(
            _convert_frontier_edges_to_fx_node_to_kwargs(graph, node_to_fx_node, subgraph.exiting)
        )
        fx_subgraph.dominating.update(
            _convert_frontier_edges_to_fx_node_to_kwargs(graph, node_to_fx_node, subgraph.dominating)
        )
        semantics_to_fx_subgraph[fx_subgraph.semantics] = fx_subgraph
    # annotate the ``fx.GraphModule``
    fx_node_to_kwarg_to_fx_subgraph: dict[fx.Node, base.fx.KWArgToSubgraph] = collections.defaultdict(dict)
    for _, fx_subgraph in semantics_to_fx_subgraph.items():
        for fx_node, kwargs in fx_subgraph.edges.items():
            kwargs_to_fx_subgraph = dict.fromkeys(kwargs, fx_subgraph)
            fx_node_to_kwarg_to_fx_subgraph[fx_node].update(kwargs_to_fx_subgraph)
    for fx_node, kwarg_to_fx_subgraph in fx_node_to_kwarg_to_fx_subgraph.items():
        base.fx.set_kwarg_to_subgraph(fx_node, kwarg_to_fx_subgraph)
    return fx_graph_module
