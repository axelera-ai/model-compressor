# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import functools
from typing import Any


def rgetattr(obj: Any, qualified_name: str) -> Any:
    """Recursively traverse an object's attributes to retrieve the desired one.

    Args:
        obj: An object.
        qualified_name: A dot-concatenated sequence of identifiers pointing to a nested attribute.

    Returns:
        The nested attribute.

    """
    return functools.reduce(getattr, [obj] + qualified_name.split("."))  # https://stackoverflow.com/a/31174427
