# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

from typing import Any

import model_compressor.api as api
import torch
import torch.fx as fx


def export(module: torch.nn.Module, input: dict[str, Any]) -> tuple[fx.GraphModule, dict[str, Any]]:
    """Trace an eager module with ``torch.export`` and import it into a demo-adaptable ``GraphModule``.

    Args:
        module: The eager module to trace.
        input: The keyword example input; its keys must match ``module.forward``'s parameter names.

    Returns:
        The imported ``GraphModule`` together with the example input that produced it.

    """
    module.eval()
    exported_program = torch.export.export(module, args=(), kwargs=input)
    graph_module = api.import_exported_program(exported_program)
    return graph_module, input
