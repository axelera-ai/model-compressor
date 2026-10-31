# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import types
from typing import Type

import pytest

import utils.profiler as profiler


@pytest.fixture(name="symbol_finder")
def _symbol_finder() -> profiler.SymbolFinder:
    """A symbol finder."""
    return profiler.SymbolFinder()


@pytest.fixture(name="module")
def _module() -> types.ModuleType:
    """A module."""
    import tests.utils.profiler.mock_module as module
    return module


@pytest.fixture(name="class_")
def _class(module: types.ModuleType) -> Type:
    """A class."""
    class_ = getattr(module, "PublicClass")
    return class_


class TestSymbolFinder:
    """Exercise the public methods of the `SymbolFinder` class."""

    def test_find_private_functions(self, symbol_finder: profiler.SymbolFinder, module: types.ModuleType) -> None:
        """Exercise the `find_private_functions` method.

        Args:
            symbol_finder: The instance whose method will be exercised.
            module: The argument for the method.

        """
        expected_symbols = {"_private_function"}
        actual_symbols = {symbol for symbol in symbol_finder.find_private_functions(module).keys()}
        assert actual_symbols == expected_symbols

    def test_find_public_functions(self, symbol_finder: profiler.SymbolFinder, module: types.ModuleType) -> None:
        """Exercise the `find_public_functions` method.

        Args:
            symbol_finder: The instance whose method will be exercised.
            module: The argument for the method.

        """
        expected_symbols = {"public_function"}
        actual_symbols = {symbol for symbol in symbol_finder.find_public_functions(module).keys()}
        assert actual_symbols == expected_symbols

    def test_find_private_classes(self, symbol_finder: profiler.SymbolFinder, module: types.ModuleType) -> None:
        """Exercise the `find_private_classes` method.

        Args:
            symbol_finder: The instance whose method will be exercised.
            module: The argument for the method.

        """
        expected_symbols = {"_PrivateClass"}
        actual_symbols = {symbol for symbol in symbol_finder.find_private_classes(module).keys()}
        assert actual_symbols == expected_symbols

    def test_find_public_classes(self, symbol_finder: profiler.SymbolFinder, module: types.ModuleType) -> None:
        """Exercise the `find_public_classes` method.

        Args:
            symbol_finder: The instance whose method will be exercised.
            module: The argument for the method.

        """
        expected_symbols = {"PublicClass"}
        actual_symbols = {symbol for symbol in symbol_finder.find_public_classes(module).keys()}
        assert actual_symbols == expected_symbols

    def test_find_private_methods(self, symbol_finder: profiler.SymbolFinder, class_: Type) -> None:
        """Exercise the `find_private_methods` method.

        Args:
            symbol_finder: The instance whose method will be exercised.
            class_: The argument for the method.

        """
        expected_symbols = {"_private_method"}
        actual_symbols = {symbol for symbol in symbol_finder.find_private_methods(class_).keys()}
        assert actual_symbols == expected_symbols

    def test_find_public_methods(self, symbol_finder: profiler.SymbolFinder, class_: Type) -> None:
        """Exercise the `find_public_methods` method.

        Args:
            symbol_finder: The instance whose method will be exercised.
            class_: The argument for the method.

        """
        expected_symbols = {"public_method"}
        actual_symbols = {symbol for symbol in symbol_finder.find_public_methods(class_).keys()}
        assert actual_symbols == expected_symbols
