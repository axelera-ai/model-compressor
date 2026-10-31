# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import types
from typing import Callable, Dict, List

import pytest

import utils.profiler as profiler
import tests.utils.profiler.mock_module as module


@pytest.fixture(name="function")
def _function() -> types.FunctionType:
    """A function."""
    function = module.public_function
    return function


@pytest.fixture(name="n_calls")
def _n_calls() -> int:
    """The number of calls to the profiled function."""
    return 2**10


@pytest.fixture(name="caller")
def _caller(function: types.FunctionType, n_calls: int) -> Callable:
    """A top-level caller function.

    Args:
        function: A function to be profiled.
        n_calls: The number of calls to the function to be profiled.

    """
    def caller(*args, **kwargs) -> None:
        for _ in range(0, n_calls):
            function()
    return caller


@pytest.fixture(name="args")
def _args() -> List[None]:
    """An empty container of positional arguments."""
    return []


@pytest.fixture(name="kwargs")
def _kwargs() -> Dict[None, None]:
    """An empty container of keyword arguments."""
    return {}


@pytest.fixture(name="expected_n_hits")
def _expected_n_hits(n_calls: int) -> int:
    """The number of times that each line of the profiled functions is expected to be hit."""
    return n_calls


@pytest.fixture(name="uninstrumented_profiler")
def _uninstrumented_profiler() -> profiler.Profiler:
    """A `Profiler` instance."""
    return profiler.Profiler()


@pytest.fixture(name="instrumented_profiler")
def _instrumented_profiler(uninstrumented_profiler: profiler.Profiler, function: types.FunctionType) -> profiler.Profiler:
    """A `Profiler` instance.

    Args:
        uninstrumented_profiler: An instance that is profiling no function.
        function: The function to be profiled.
    
    Returns:
        An instance that is profiling a function.

    """
    uninstrumented_profiler.instrument(function)
    return uninstrumented_profiler


class TestProfiler:
    """Exercise the public methods of the `Profiler` class."""

    def test___init__(self) -> None:
        """Exercise the initializer method."""
        profiler_ = profiler.Profiler()
        assert profiler_.functions == []

    def test_instrument(self, uninstrumented_profiler: profiler.Profiler, function: types.FunctionType) -> None:
        """Exercise the `instrument` method.

        Args:
            uninstrumented_profiler: The instance whose method is exercised.
            function: The argument to the method.

        """
        uninstrumented_profiler.instrument(function)
        assert uninstrumented_profiler.functions == [function]

    def test_uninstrument(self, instrumented_profiler: profiler.Profiler, function: types.FunctionType) -> None:
        """Exercise the `uninstrument` method.

        Args:
            instrumented_profiler: The instance whose method is exercised.
            function: The argument to the method.

        """
        instrumented_profiler.uninstrument(function)
        assert instrumented_profiler.functions == []

    def test_run(self, instrumented_profiler: profiler.Profiler, caller: Callable, args: List[None], kwargs: Dict[None, None], expected_n_hits: int) -> None:
        """Exercise the `run` method.

        Args:
            instrumented_profiler: The instance whose method is exercised.
            caller: The first argument to the method.
            args: The second argument to the method.
            kwargs: The third argument to the method.
            expected_n_hits: The expected number of hits for each line of the profiled functions.

        """
        output, statistics = instrumented_profiler.run(caller, *args, **kwargs)
        assert output == caller(*args, **kwargs)
        assert {function.__name__ for function in instrumented_profiler.functions} == {name for _, _, name in statistics.timings.keys()}
        assert all((actual_n_hits == expected_n_hits) for timings in statistics.timings.values() for _, actual_n_hits, _ in timings)
