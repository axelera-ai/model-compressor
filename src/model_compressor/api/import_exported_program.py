# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import copy
import functools
import inspect
import operator
import types
from collections.abc import Callable
from typing import cast

import torch
import torch._ops
import torch.fx as fx
import torch.nn as nn
from packaging import version

import model_compressor.ops.opset as opset
import model_compressor.passes.utils.normalize as normalize


@functools.cache
def _get_decompositions(whitelist: tuple[opset.Operator]) -> dict[opset.Operator, Callable]:
    """Get the decompositions to ATen Core operators for non-whitelisted operators.

    !!! note

        The function output is cached according to the output since it is expected to be called for selected opsets.

    Args:
        whitelist: The collection of operators to be guarded against decomposition.

    Returns:
        The decompositions to ATen Core operators for non-whitelisted operators.

    """
    decompositions = cast(dict[opset.Operator, Callable], copy.copy(torch.export.default_decompositions()))
    for op in set(whitelist) - opset.OPSET_ATEN_CORE:
        if op in decompositions:
            del decompositions[op]
    return decompositions


def _has_side_effect(node: fx.Node) -> bool:
    """Verify whether a node of a graph PyTorch model with no successors has a side-effect.

    Args:
        node: A node of a graph PyTorch model.

    Returns:
        Whether the node has a side-effect - or it can't be ruled out that it has.

    """
    if node.op == "placeholder":
        return False  # if unused, a model input has no side-effect
    if node.op == "get_attr":
        return False  # if unused, a model buffer/parameter has no side-effect
    if node.op == "output":
        return True  # the model must return an output
    # `node.op == "call_function"`
    is_pure_operator = inspect.isbuiltin(node.target) and (node.target in (operator.eq, operator.getitem))
    is_pure_op_overload = (
        isinstance(node.target, torch._ops.OpOverload)
        and hasattr(node.target, "_schema")
        and (not node.target._schema.is_mutable)
    )
    return not (is_pure_operator or is_pure_op_overload)


def import_exported_program(exported_program: torch.export.ExportedProgram) -> fx.GraphModule:
    """Import a PyTorch ExportedProgram by converting it into a graph PyTorch GraphModule
    composed solely of whitelisted operators. The ExportedProgram can be obtained by torch.export.export.

    Args:
        exported_program: The ExportedProgram obtained by torch.export.export.

    Returns:
        A functionally equivalent graph PyTorch model composed solely of whitelisted operators.
    """
    # freeze the opset
    decompositions = _get_decompositions(tuple(opset.OPSET))

    # export the model to a graph PyTorch model
    # torch 2.9 introduced shape guards, we disable them to avoid handling those guards
    kwargs = {"check_guards": False} if (version.parse(torch.__version__) >= version.parse("2.9.0")) else {}
    model = cast(fx.GraphModule, exported_program.run_decompositions(decompositions).module(**kwargs))
    model.graph.eliminate_dead_code(lambda node: _has_side_effect(node))
    model.graph.lint()
    model.recompile()
    # an `ExportedProgram` overwrites `Module.train` and `Module.eval` methods, so we must restore the original behaviour
    model.train = types.MethodType(nn.Module.train, model)
    model.eval = types.MethodType(nn.Module.eval, model)
    model.eval()
    # ensure that all `Node`s use only keyword arguments
    model = normalize.Normalize()(model).graph_module
    return model
