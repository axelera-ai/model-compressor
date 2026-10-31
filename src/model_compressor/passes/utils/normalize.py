# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import cast

import torch
import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base


class Normalize(pass_base.PassBase):
    """Normalize `GraphModule`s so that their non-builtin `'call_function'` nodes use keyword-only arguments."""

    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        """Normalize a `GraphModule` so that its non-builtin `'call_function'` nodes use keyword-only arguments.

        Args:
            graph_module: A `GraphModule`.

        Returns:
            A pair whose first item is the normalized `GraphModule`, and whose second item is a constant `True` flag.

        Raises:
            ValueError: A node expects keyword arguments that are not provided.

        """
        for node in graph_module.graph.nodes:
            if (node.op == "call_function") and isinstance(node.target, torch._ops.OpOverload):
                maybe_normalized_arguments = node.normalized_arguments(graph_module, normalize_to_only_use_kwargs=True)
                if maybe_normalized_arguments is not None:
                    node.args, node.kwargs = (), maybe_normalized_arguments.kwargs
        for node in graph_module.graph.nodes:
            if (node.op == "call_function") and isinstance(node.target, torch._ops.OpOverload):
                schema = cast(torch.FunctionSchema, node.target._schema)
                expected_names = {argument.name for argument in schema.arguments}
                actual_names = set(node.kwargs)
                if actual_names == expected_names:
                    continue
                underspecified_names = expected_names - actual_names
                overspecified_names = actual_names - expected_names
                if (overspecified_names == {"input"}) and (underspecified_names == {"self"}):
                    kwargs = {
                        argument.name: node.kwargs[(argument.name if (argument.name != "self") else "input")]
                        for argument in schema.arguments
                    }
                    node.kwargs = kwargs
                else:
                    raise ValueError(
                        f"Node with target {node.target} expects values for parameters {underspecified_names}"
                    )
        graph_module.recompile()
        return pass_base.PassResult(graph_module, True)


class Validate(pass_base.PassBase):
    """Validate that `GraphModule`s are well-formed - i.e., that they contain only nodes with legal opcodes and normalized."""

    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        """Validate that a `GraphModule` is well-formed - i.e., it contains only nodes with legal opcodes and normalized.

        Args:
            graph_module: A `GraphModule`.

        Returns:
            A pair whose first item is the (valid) `GraphModule`, and whose second item is a constant `False` flag.

        Raises:
            ValueError: The `GraphModule` contains nodes with illegal opcodes.
            ValueError: The `GraphModule` contains non-builtin `'call_function'` nodes with positional arguments.

        """
        # check that the `GraphModule` contains only nodes with legal opcodes
        legal_opcodes = ("placeholder", "output", "get_attr", "call_function")
        illegal_opcode_nodes = {node for node in graph_module.graph.nodes if (node.op not in legal_opcodes)}
        if illegal_opcode_nodes:
            raise ValueError(f"GraphModule must contain only nodes with legal opcodes {legal_opcodes}")
        # check that the `GraphModule` contains only non-builtin `'call_function'` nodes with keyword arguments
        positional_argument_nodes = {
            node
            for node in graph_module.graph.nodes
            if (node.op == "call_function") and isinstance(node.target, torch._ops.OpOverload) and (len(node.args) > 0)
        }
        if positional_argument_nodes:
            raise ValueError(
                f"GraphModule must contain only non-builtin {'call_function'!r} nodes with keyword arguments"
            )
        return pass_base.PassResult(graph_module, False)
