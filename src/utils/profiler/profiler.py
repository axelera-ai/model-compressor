# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import types
from typing import Any, Callable, List, Set, Tuple

import line_profiler

LineStats = getattr(getattr(line_profiler, "_line_profiler"), "LineStats")
"""A mapping from functions to their per-line profiling statistics."""


class Profiler:
    """Profile functions, per-line."""

    def __init__(self) -> None:
        """Initialize the profiler."""
        self._functions: Set[types.FunctionType] = set()

    @property
    def functions(self) -> List[types.FunctionType]:
        return sorted(self._functions, key=lambda function: function.__qualname__)

    def instrument(self, function: types.FunctionType) -> None:
        """Enable per-line profiling for a target function.

        Args:
            function: The function that should be profiled, per-line.

        """
        self._functions.add(function)

    def uninstrument(self, function: types.FunctionType) -> None:
        """Disable per-line profiling for a target function.

        Args:
            function: The function that should no longer per profiled.

        """
        if function in self._functions:
            self._functions.remove(function)

    def run(self, caller: Callable, *args, **kwargs) -> Tuple[Any, LineStats]:
        """Collect per-line profiling statistics about the instrumented functions, in the context of a top-level call.

        Args:
            caller: The top-level caller function.
            args: The positional arguments for the top-level caller function.
            kwargs: The keyword arguments for the top-level caller function.

        Returns:
            A pair whose items have the following semantics:
                * the first item is the output of the top-level caller function;
                * the second argument are the per-line profiling statistics about the instrumented functions.

        """
        profiler = line_profiler.LineProfiler()
        for function in self._functions:
            profiler.add_function(function)
        caller = profiler.wrap_function(caller)
        output = caller(*args, **kwargs)
        statistics = profiler.get_stats()
        return output, statistics
