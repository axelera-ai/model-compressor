# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import builtins
import copy
import functools
import operator
from typing import Any, Literal, cast

import networkx as nx
import torch.fx as fx
import torch.fx.immutable_collections
import torch.nn as nn

import model_compressor.passes.adapt.base as base
import model_compressor.passes.annotate as annotate
import model_compressor.passes.annotate.choose_schemas as choose_schemas


class _Stub:
    """A sentinel flagging ``fx.Node`` arguments."""


def unzip_kwargs(
    kwargs: torch.fx.immutable_collections.immutable_dict,
) -> tuple[dict[str, Any], dict[base.nx.Port, fx.Node]]:
    """Separate ``fx.Node`` arguments to an ``fx.Node`` from its non-``fx.Node`` ones.

    Args:
        kwargs: The keyword arguments to an ``fx.Node``.

    Returns:
        A pair whose items have the following semantics.
            * A copy of the keyword arguments where ``fx.Node`` items have been stubbed with
              sentinel ``_Stub`` values.
            * A mapping from ``fx.Node`` argument paths to ``fx.Node`` instances.

    """
    # traverse and stub non-``fx.Node`` arguments
    stubbed: dict[str, Any] = {}
    for key, value in kwargs.items():
        match type(value):
            case torch.fx.immutable_collections.immutable_list:
                stubbed[key] = [_Stub() if isinstance(item, fx.Node) else item for item in value]
            case _:
                assert not isinstance(value, (tuple, torch.fx.immutable_collections.immutable_dict))
                stubbed[key] = _Stub() if isinstance(value, fx.Node) else value
    # traverse and register ``fx.Node`` arguments
    patch: dict[base.nx.Port, fx.Node] = {}
    for key, value in kwargs.items():
        match type(value):
            case torch.fx.immutable_collections.immutable_list:
                for idx, item in enumerate(value):
                    if isinstance(item, fx.Node):
                        patch[(key, idx)] = item
            case _:
                if isinstance(value, fx.Node):
                    patch[(key,)] = value
    return stubbed, patch


def zip_kwargs(
    stubbed: dict[str, Any], patch: dict[base.nx.Port, fx.Node]
) -> torch.fx.immutable_collections.immutable_dict:
    """Merge the ``fx.Node`` arguments to an ``fx.Node`` with its non-``fx.Node`` ones.

    Args:
        stubbed: A copy of the keyword arguments where ``fx.Node`` items have been stubbed with
            sentinel ``_Stub`` values.
        patch: A mapping from ``fx.Node`` argument paths to ``fx.Node`` instances.

    Returns:
        The keyword arguments to an ``fx.Node``.

    """
    kwargs = copy.deepcopy(stubbed)
    for port, node in patch.items():
        head, *path = port
        if len(path) == 0:
            key = head
            kwargs[key] = node
        else:
            key, (idx,) = head, path
            kwargs[key][idx] = node
    # make all containers immutable
    for key, value in kwargs.items():
        if isinstance(value, builtins.list):
            kwargs[key] = torch.fx.immutable_collections.immutable_list(value)
    # ensure that all ``_Stub`` leaves have been patched
    for _, value in kwargs.items():
        match type(value):
            case torch.fx.immutable_collections.immutable_list:
                assert not any(isinstance(item, _Stub) for item in value)
            case _:
                assert not isinstance(value, (tuple, dict))
                assert not isinstance(value, _Stub)
    return torch.fx.immutable_collections.immutable_dict(kwargs)


def convert_fx_to_nx(fx_graph_module: fx.GraphModule) -> base.nx.Graph:
    """Convert an ``fx.GraphModule`` to a ``networkx.MultiDiGraph``.

    Args:
        graph_module: A source ``fx.GraphModule``.

    Returns:
        An equivalent ``networkx.MultiDiGraph``.

    """
    # canonicalize kwargs
    fx_node_to_stubbed_and_patch: dict[fx.Node, tuple[dict[str, Any], dict[base.nx.Port, fx.Node]]] = {}
    for fx_node in fx_graph_module.graph.nodes:
        match fx_node.op:
            case "placeholder" | "get_attr":
                args, kwargs = fx_node.args, fx_node.kwargs
                assert args == ()
            case "output":
                args, kwargs = fx_node.args, fx_node.kwargs
                assert kwargs == {}
                (arg,) = args
                kwargs_: dict[base.nx.Port, fx.Node]
                match type(arg):
                    case builtins.tuple:
                        kwargs_ = {("", idx): arg for idx, arg in enumerate(arg)}
                    case fx.Node:
                        kwargs_ = {("",): arg}
                    case _:
                        raise TypeError(f"Output has illegal type {type(arg)}")
                kwargs = torch.fx.immutable_collections.immutable_dict(kwargs_)
            case "call_function":
                if fx_node.target is operator.getitem:
                    continue
                args, kwargs = fx_node.args, fx_node.kwargs
                assert args == (), "Graph must be keyword-only normalized"
            case _:
                raise ValueError(f"Node {fx_node} has illegal opcode {fx_node.op}")
        stubbed, patch = unzip_kwargs(kwargs)
        fx_node_to_stubbed_and_patch[fx_node] = stubbed, patch
    # create the ``base.Graph``
    graph = base.nx.Graph()
    for fx_node, (_, patch) in fx_node_to_stubbed_and_patch.items():
        graph.add_node(fx_node.name)
        output_port_pred: base.nx.Port
        for input_port, fx_pred in patch.items():
            if (fx_pred.op == "call_function") and (fx_pred.target is operator.getitem):
                fx_getitem = fx_pred
                fx_pred, idx = cast(tuple[fx.Node, int], fx_getitem.args)
                output_port_pred = (fx_getitem.name, idx)  # prefix the port with ``"getitem"``
            else:
                output_port_pred = ("",)
            graph.add_edge(fx_pred.name, fx_node.name, key=(output_port_pred, input_port))
    # The ``base.Graph`` is ready: transfer metadata.
    # annotate ``base.Node``s
    for fx_node in fx_graph_module.graph.nodes:
        op: Literal["placeholder", "get_attr", "output", "call_function"]
        match fx_node.op:
            case "placeholder":
                op = fx_node.op
                target = fx_node.target
                stubbed, _ = fx_node_to_stubbed_and_patch[fx_node]
                value = None
                fx_annotation = annotate.get_annotation(fx_node)
                assert fx_annotation.source is None
                assert fx_annotation.chosen is None
            case "get_attr":
                op = fx_node.op
                target = fx_node.target
                stubbed, _ = fx_node_to_stubbed_and_patch[fx_node]
                attr = functools.reduce(getattr, (fx_graph_module, *fx_node.target.split(".")))
                assert isinstance(attr, (torch.Tensor, nn.Parameter)), (
                    f"Node {fx_node} points to attribute with illegal type {type(attr)}"
                )
                with torch.no_grad():
                    value = type(attr)(attr.detach().clone())
                fx_annotation = annotate.get_annotation(fx_node)
                assert fx_annotation.source is None
                assert fx_annotation.chosen is None
            case "output":
                op = fx_node.op
                target = fx_node.target
                stubbed, _ = fx_node_to_stubbed_and_patch[fx_node]
                value = None
                fx_annotation = annotate.get_annotation(fx_node)
                assert fx_annotation.source is None
                assert fx_annotation.chosen is None
            case "call_function":
                if fx_node.target is operator.getitem:
                    continue
                op = fx_node.op
                target = fx_node.target
                stubbed, _ = fx_node_to_stubbed_and_patch[fx_node]
                value = None
                fx_annotation = annotate.get_annotation(fx_node)
                assert fx_annotation.source is not None
                assert fx_annotation.chosen is not None
            case _:
                raise ValueError(f"Node {fx_node} has illegal opcode {fx_node.op}")
        annotation = base.nx.Annotation(op=op, target=target, kwargs=stubbed, value=value, annotation=fx_annotation)
        base.nx.set_annotation(graph, fx_node.name, annotation)
    return graph


def convert_nx_to_fx(graph: base.nx.Graph) -> fx.GraphModule:
    """Convert a ``networkx.MultiDiGraph`` into an ``fx.GraphModule``.

    Args:
        graph: A source ``networkx.MultiDiGraph``.

    Returns:
        An equivalent ``fx.GraphModule``.

    """
    args: tuple[Any, ...]
    kwargs: torch.fx.immutable_collections.immutable_dict
    # create the ``fx.GraphModule``
    node_to_fx_node: dict[base.nx.Node, fx.Node] = {}
    nn_module, fx_graph = nn.Module(), fx.Graph()
    for node in nx.topological_sort(graph):
        annotation = base.nx.get_annotation(graph, node)
        # register parameters/buffers in the ``nn.Module``
        match annotation.op:
            case "placeholder" | "output" | "call_function":
                pass
            case "get_attr":
                *path, target = cast(str, annotation.target).split(".")
                nn_parent = nn_module
                for nn_child_name in path:
                    if not hasattr(nn_parent, nn_child_name):
                        nn_child = nn.Module()
                        nn_parent.register_module(nn_child_name, nn_child)
                    nn_parent = getattr(nn_parent, nn_child_name)
                match type(annotation.value):
                    case nn.Parameter:
                        nn_parent.register_parameter(target, cast(nn.Parameter, annotation.value))
                    case torch.Tensor:
                        nn_parent.register_buffer(target, annotation.value)
                    case _:
                        raise TypeError(
                            f"Attribute must have type {(nn.Parameter, torch.Tensor)}, has type {type(annotation.value)}"
                        )
            case _:
                raise ValueError(f"Node {node} has illegal opcode {annotation.op}")
        # create the ``fx.Node`` argument patch from incoming edges
        patch = {}
        for pred, _, (output_port_pred, input_port) in graph.in_edges(node, keys=True):
            head, *path = output_port_pred
            if len(path) == 0:
                fx_pred = node_to_fx_node[pred]
                patch[input_port] = fx_pred
            else:
                assert len(path) == 1
                getitem, (idx_,) = head, path
                maybe_getitem_fx_pred = [
                    fx_pred
                    for fx_pred in node_to_fx_node[pred].users
                    if (
                        (fx_pred.op == "call_function")
                        and (fx_pred.target is operator.getitem)
                        and (fx_pred.args[1] == idx_)
                    )
                ]
                assert len(maybe_getitem_fx_pred) in (0, 1)
                if len(maybe_getitem_fx_pred) == 0:
                    args, kwargs = (node_to_fx_node[pred], idx_), torch.fx.immutable_collections.immutable_dict({})
                    fx_pred = fx_graph.create_node(
                        op="call_function", target=operator.getitem, args=args, kwargs=kwargs, name=getitem
                    )
                else:
                    fx_pred = next(iter(maybe_getitem_fx_pred))
                patch[input_port] = fx_pred
        # prepare the ``fx.Node``'s args/kwargs
        match annotation.op:
            case "placeholder" | "get_attr" | "call_function":
                args, kwargs = (), zip_kwargs(annotation.kwargs, patch)
            case "output":
                args, kwargs = (), zip_kwargs(annotation.kwargs, patch)
                if (len(kwargs) == 1) and (len(next(iter(kwargs))) == 1):
                    assert next(iter(kwargs.keys())) == ("",)
                    arg = next(iter(kwargs.values()))
                else:
                    arg = tuple((value for _, value in sorted(kwargs.items(), key=lambda item: item[0][1])))
                args, kwargs = (arg,), torch.fx.immutable_collections.immutable_dict()
            case _:
                raise ValueError(f"Node {node} has illegal opcode {annotation.op}")
        fx_node = fx_graph.create_node(op=annotation.op, target=annotation.target, args=args, kwargs=kwargs, name=node)
        node_to_fx_node[node] = fx_node
    fx_graph_module = fx.GraphModule(root=nn_module, graph=fx_graph)
    fx_graph_module.graph.lint()
    fx_graph_module.recompile()
    # The ``fx.GraphModule`` is ready: transfer metadata.
    # annotate ``fx.Node``s
    fx_node_to_node: dict[fx.Node, base.nx.Node] = {fx_node: node for node, fx_node in node_to_fx_node.items()}
    for fx_node in fx_graph_module.graph.nodes:
        match fx_node.op:
            case "placeholder" | "get_attr" | "output":
                fx_annotation = base.nx.get_annotation(graph, fx_node_to_node[fx_node]).annotation
                assert fx_annotation.source is None
                assert fx_annotation.chosen is None
            case "call_function":
                if fx_node.target is operator.getitem:
                    # ``operator.getitem`` nodes are elided when an ``fx.GraphModule`` is converted to a
                    # ``networkx.MultiDiGraph`` (they are absorbed into a producer's output port), so the
                    # annotations created here are destroyed if another FX-to-NX conversion is performed:
                    # they are ephemeral, and cannot be expected to persist.
                    fx_pred, idx = cast(tuple[fx.Node, int], fx_node.args)
                    fx_annotation = annotate.get_annotation(fx_pred)
                    assert fx_annotation.source is not None
                    assert fx_annotation.chosen is not None
                    fx_annotation = annotate.Annotation(
                        source=choose_schemas.synthesize_schema_for_operator_getitem(fx_annotation.source, idx),
                        chosen=choose_schemas.synthesize_schema_for_operator_getitem(fx_annotation.chosen, idx),
                    )
                else:
                    fx_annotation = base.nx.get_annotation(graph, fx_node_to_node[fx_node]).annotation
                    assert fx_annotation.source is not None
                    assert fx_annotation.chosen is not None
            case _:
                raise ValueError(f"Node {fx_node} has illegal opcode {fx_node.op}")
        annotate.set_annotation(fx_node, fx_annotation)
    return fx_graph_module
