# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import bisect
import collections
import dataclasses
import operator
from collections.abc import Callable

import torch
import torch.fx as fx

_UNKNOWN = None


@dataclasses.dataclass
class _Ids:
    booked: list[int] = dataclasses.field(default_factory=list)
    free: list[int] = dataclasses.field(default_factory=list)


class State:
    """Track the available names for functional nodes in an ``fx.Graph``."""

    def __init__(self, graph: fx.Graph) -> None:
        self._target_to_ids: dict[Callable, _Ids] = collections.defaultdict(_Ids)
        for node in graph.nodes:
            if node.op == "call_function":
                id = self._get_id(node.target, node.name)
                if id is not _UNKNOWN:
                    self._target_to_ids[node.target].booked.append(id)
        for target, ids in self._target_to_ids.items():
            ids = _Ids(
                booked=sorted(ids.booked),
                free=sorted(set(range(0, max(ids.booked) + 1)) - set(ids.booked)),
            )
            self._target_to_ids[target] = ids

    def _get_target_name(self, target: Callable) -> str:
        target_name: str
        if isinstance(target, torch._ops.OpOverload):
            *_, target_name = target.name().split("::")
        else:
            if target is operator.getitem:
                target_name = "getitem"
            else:
                raise ValueError(f"Unknown target {target}")
        return target_name

    def _get_id(self, target: Callable, name: str) -> int | None:
        id: int | None
        target_name = self._get_target_name(target)
        maybe_id = name.removeprefix(target_name).removeprefix("_")
        if maybe_id == "":
            id = 0
        elif maybe_id.isdecimal():
            id = int(maybe_id)
        else:
            id = _UNKNOWN
        return id

    def free(self, target: Callable, name: str) -> None:
        ids = self._target_to_ids[target]
        id = self._get_id(target, name)
        if (id is not _UNKNOWN) and (id in ids.booked):
            ids.booked.remove(id)
            if id not in ids.free:
                bisect.insort(ids.free, id)

    def book(self, target: Callable) -> str:
        ids = self._target_to_ids[target]
        if len(ids.free) == 0:
            id = 0 if (len(ids.booked) == 0) else (ids.booked[-1] + 1)
            bisect.insort(ids.free, id)
        id = ids.free.pop(0)
        bisect.insort(ids.booked, id)
        target_name = self._get_target_name(target)
        name = f"{target_name}_{id}" if (id > 0) else f"{target_name}"
        return name
