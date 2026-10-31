# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

"""Map operator schemas to identifier-safe, human-readable names for selection."""

import builtins
import types
import typing
from collections.abc import Set
from typing import Any, get_args, get_origin

import model_compressor.database.schema as _schema


def _stringify(type_: Any) -> str:
    match get_origin(type_):
        case builtins.list:
            type_, *_ = get_args(type_)
            mnemonic = f"l{_stringify(type_)}"
        case typing.Literal:
            arg, *_ = get_args(type_)
            mnemonic = arg.name.lower()
    return mnemonic


def _mangle(schema: _schema.Schema) -> str:
    engine = schema.engine.name.lower()
    target = schema.target.name().replace("::", "_").replace(".", "_")
    input = "".join([_stringify(argument.type) for argument in schema.signature.input])
    output = "".join([_stringify(argument.type) for argument in schema.signature.output])
    name = "_".join([engine, target, input, output])
    return name


class Catalog(types.SimpleNamespace):
    def __init__(self, database: Set[_schema.Schema]) -> None:
        name_to_schema: dict[str, _schema.Schema] = {}
        for schema in database:
            name = _mangle(schema)
            if name in name_to_schema:
                raise ValueError(f"Name collision: {name!r} is shared by two schemas")
            name_to_schema[name] = schema
        super().__init__(**name_to_schema)
