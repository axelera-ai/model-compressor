# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch
import torch.fx as fx
import torch.fx.passes.infra.pass_base as pass_base

import model_compressor.nn as nn
import model_compressor.ops  # noqa: F401 - register the xaten custom operators
import model_compressor.passes.annotate as annotate

_OP_OVERLOAD_TO_INPUT_NAME_AND_OTHER_ARGUMENT_NAMES: dict[torch._ops.OpOverload, tuple[str, list[str]]] = {
    torch.ops.aten.silu.default: ("self", []),
}
"""Map `OpOverload` functional operators to the pair whose items have the following semantics:

* the first item is the name of the argument that is exposed also by the `OptimizingActivation` module;
* the second item is the name of the remaining arguments, that are assumed to be hardcoded into the functional
  operator node (i.e., none of them is an FX node).

"""


class ModularizeIntoOptimizingActivation(pass_base.PassBase):
    """Convert each supported activation functional operator into an `OptimizingActivation` modular operation."""

    def call(self, graph_module: fx.GraphModule) -> pass_base.PassResult:
        # find the matches
        matches = []
        for node in graph_module.graph.nodes:
            if (node.op == "call_function") and (node.target in _OP_OVERLOAD_TO_INPUT_NAME_AND_OTHER_ARGUMENT_NAMES):
                _, other_argument_names = _OP_OVERLOAD_TO_INPUT_NAME_AND_OTHER_ARGUMENT_NAMES[node.target]
                if not any(isinstance(node.kwargs[name], fx.Node) for name in other_argument_names):
                    matches.append((node,))
        # rewrite the matches
        if len(matches) == 0:  # nothing to rewrite: exit early
            return pass_base.PassResult(graph_module, False)
        for match in matches:
            (old_node,) = match
            # create the modular operation
            input_name, other_argument_names = _OP_OVERLOAD_TO_INPUT_NAME_AND_OTHER_ARGUMENT_NAMES[old_node.target]
            new_module = nn.OptimizingActivation(
                op=old_node.target,
                name_to_parameter={name: old_node.kwargs[name] for name in other_argument_names},
            )
            # insert the modular operation node
            new_target = f"{old_node.name}_optimizing_module"
            graph_module.register_module(new_target, new_module)
            new_kwargs = {"input": old_node.kwargs[input_name]}
            with graph_module.graph.inserting_before(old_node):
                new_node = graph_module.graph.call_module(new_target, args=(), kwargs=new_kwargs)
            # remove the functional operator node
            old_node.replace_all_uses_with(new_node)
            if len(old_node.users) == 0:
                graph_module.graph.erase_node(old_node)
            # annotate the modular operation node
            old_annotation = annotate.get_annotation(old_node)
            new_annotation = annotate.Annotation(
                source=old_annotation.chosen,
                chosen=old_annotation.chosen,
            )
            annotate.set_annotation(new_node, new_annotation)
        # consolidate the changes
        graph_module.graph.lint()
        graph_module.recompile()
        return pass_base.PassResult(graph_module, True)
