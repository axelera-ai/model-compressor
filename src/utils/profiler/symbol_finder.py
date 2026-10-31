# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import inspect
import types
from typing import Dict, Type

from typing_extensions import Annotated, TypeAlias

Symbol: TypeAlias = Annotated[str, lambda value: value.isidentifier()]
"""An identifier."""

UnboundMethod: TypeAlias = types.FunctionType
"""An unbound method of a class."""  # https://stackoverflow.com/a/3228931


class SymbolFinder:
    """Find the callables in a Python codebase."""

    def _find_functions(self, module: types.ModuleType) -> Dict[Symbol, types.FunctionType]:
        """Find the functions defined by a package or module.

        Args:
            module: A package or module.

        Returns:
            A mapping from symbols to functions.

        """
        symbol_to_value = {symbol: getattr(module, symbol) for symbol in dir(module)}
        symbol_to_function = {symbol: value for symbol, value in symbol_to_value.items() if inspect.isfunction(value)}
        return symbol_to_function

    def _find_classes(self, module: types.ModuleType) -> Dict[Symbol, Type]:
        """Find the classes defined by a package or module.

        Args:
            module: A package or module.

        Returns:
            A mapping from symbols to classes.

        """
        symbol_to_value = {symbol: getattr(module, symbol) for symbol in dir(module)}
        symbol_to_class = {symbol: value for symbol, value in symbol_to_value.items() if inspect.isclass(value)}
        return symbol_to_class

    def _find_methods(self, class_: Type) -> Dict[Symbol, UnboundMethod]:
        """Find the methods defined by a class.

        Args:
            class_: A class.

        Returns:
            A mapping from symbols to methods.

        """
        symbol_to_value = {symbol: getattr(class_, symbol) for symbol in dir(class_)}
        symbol_to_method = {symbol: value for symbol, value in symbol_to_value.items() if inspect.isfunction(value)}  # https://stackoverflow.com/a/3228931
        return symbol_to_method

    def _is_public_symbol(self, symbol: Symbol) -> bool:
        """Check whether a symbol is public.

        Args:
            symbol: A symbol.

        Returns:
            Whether the symbol is public.

        """
        return not symbol.startswith("_")

    def find_private_functions(self, module: types.ModuleType) -> Dict[Symbol, types.FunctionType]:
        """Find the private functions defined by a package or module.

        Args:
            module: A package or module.

        Returns:
            A mapping from private symbols to functions.

        """
        symbol_to_function = self._find_functions(module)
        symbol_to_function = {symbol: function for symbol, function in symbol_to_function.items() if (not self._is_public_symbol(symbol))}
        return symbol_to_function

    def find_public_functions(self, module: types.ModuleType) -> Dict[Symbol, types.FunctionType]:
        """Find the public functions defined by a package or module.

        Args:
            module: A package or module.

        Returns:
            A mapping from public symbols to functions.

        """
        symbol_to_function = self._find_functions(module)
        symbol_to_function = {symbol: function for symbol, function in symbol_to_function.items() if self._is_public_symbol(symbol)}
        return symbol_to_function

    def find_private_classes(self, module: types.ModuleType) -> Dict[Symbol, Type]:
        """Find the private classes defined by a package or module.

        Args:
            module: A package or module.

        Returns:
            A mapping from private symbols to classes.

        """
        symbol_to_class = self._find_classes(module)
        symbol_to_class = {symbol: class_ for symbol, class_ in symbol_to_class.items() if (not self._is_public_symbol(symbol))}
        return symbol_to_class

    def find_public_classes(self, module: types.ModuleType) -> Dict[Symbol, Type]:
        """Find the public classes defined by a package or module.

        Args:
            module: A package or module.

        Returns:
            A mapping from public symbols to classes.

        """
        symbol_to_class = self._find_classes(module)
        symbol_to_class = {symbol: class_ for symbol, class_ in symbol_to_class.items() if self._is_public_symbol(symbol)}
        return symbol_to_class

    def find_private_methods(self, class_: Type) -> Dict[Symbol, UnboundMethod]:
        """Find the private methods defined by a class.

        Args:
            class_: A class.

        Returns:
            A mapping from private symbols to methods.

        """
        symbol_to_method = self._find_methods(class_)
        symbol_to_method = {symbol: method for symbol, method in symbol_to_method.items() if (not self._is_public_symbol(symbol))}
        return symbol_to_method

    def find_public_methods(self, class_: Type) -> Dict[Symbol, UnboundMethod]:
        """Find the public methods defined by a class.

        Args:
            class_: A class.

        Returns:
            A mapping from public symbols to methods.

        """
        symbol_to_method = self._find_methods(class_)
        symbol_to_method = {symbol: method for symbol, method in symbol_to_method.items() if self._is_public_symbol(symbol)}
        return symbol_to_method
