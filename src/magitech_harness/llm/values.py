"""Immutable value helpers and lossless JSON utilities for the LLM runtime."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NoReturn, TypeGuard

from magitech_harness.agent.cancellation import AbortSignal

MAX_SAFE_INTEGER = 2**53 - 1

_SENTINEL_INVALID = object()


def is_safe_integer(value: Any) -> TypeGuard[int]:
    """Return True if *value* is a non-boolean integer within JS safe range."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER
    )


def is_non_negative_safe_integer(value: Any) -> TypeGuard[int]:
    """Return True if *value* is a non-negative safe integer."""
    return is_safe_integer(value) and value >= 0


def assert_never(value: Any, context: str | None = None) -> NoReturn:
    """Raise ValueError when an unreachable union variant arrives at runtime."""
    try:
        rendered = json.dumps(value)
    except (TypeError, ValueError, OverflowError):
        rendered = str(value)
    scope = f" in {context}" if context else ""
    raise ValueError(f"unreachable variant{scope}: {rendered}")


def _deny_mutation(kind: str) -> NoReturn:
    """Raise TypeError for attempted mutations on frozen containers."""
    raise TypeError(f"Cannot modify frozen {kind}")


class FrozenDict(dict[Any, Any]):
    """Read-only dictionary supporting structural equality, JSON, and weakrefs."""

    def __init__(self, initial: Mapping[Any, Any] | None = None) -> None:
        super().__init__(initial or {})

    def __setitem__(self, key: Any, value: Any) -> NoReturn:
        _deny_mutation("dict")

    def __delitem__(self, key: Any) -> NoReturn:
        _deny_mutation("dict")

    def pop(self, *_args: Any, **_kwargs: Any) -> NoReturn:
        _deny_mutation("dict")

    def popitem(self) -> NoReturn:
        _deny_mutation("dict")

    def clear(self) -> NoReturn:
        _deny_mutation("dict")

    def update(self, *_args: Any, **_kwargs: Any) -> NoReturn:
        _deny_mutation("dict")

    def setdefault(self, *_args: Any, **_kwargs: Any) -> NoReturn:
        _deny_mutation("dict")

    def __ior__(self, _other: Any) -> NoReturn:  # type: ignore[misc]
        _deny_mutation("dict")

    def __copy__(self) -> dict[Any, Any]:
        return dict(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[Any, Any]:
        cloned: dict[Any, Any] = {}
        memo[id(self)] = cloned
        for key, item in self.items():
            cloned[copy.deepcopy(key, memo)] = copy.deepcopy(item, memo)
        return cloned


class FrozenList(list[Any]):
    """Read-only list supporting structural equality, JSON, and weakrefs."""

    def __init__(self, iterable: Sequence[Any] = ()) -> None:
        super().__init__(iterable)

    def __setitem__(self, index: Any, value: Any) -> NoReturn:
        _deny_mutation("list")

    def __delitem__(self, index: Any) -> NoReturn:
        _deny_mutation("list")

    def append(self, _item: Any) -> NoReturn:
        _deny_mutation("list")

    def extend(self, _iterable: Any) -> NoReturn:
        _deny_mutation("list")

    def insert(self, _index: Any, _item: Any) -> NoReturn:
        _deny_mutation("list")

    def pop(self, *_args: Any) -> NoReturn:
        _deny_mutation("list")

    def remove(self, _item: Any) -> NoReturn:
        _deny_mutation("list")

    def clear(self) -> NoReturn:
        _deny_mutation("list")

    def sort(self, *_args: Any, **_kwargs: Any) -> NoReturn:
        _deny_mutation("list")

    def reverse(self) -> NoReturn:
        _deny_mutation("list")

    def __iadd__(self, _other: Any) -> NoReturn:  # type: ignore[misc]
        _deny_mutation("list")

    def __imul__(self, _other: Any) -> NoReturn:  # type: ignore[misc]
        _deny_mutation("list")

    def __copy__(self) -> list[Any]:
        return list(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        cloned: list[Any] = []
        memo[id(self)] = cloned
        cloned.extend(copy.deepcopy(item, memo) for item in self)
        return cloned


def is_frozen(value: Any) -> bool:
    """Return True if *value* is an immutable container or tuple."""
    return isinstance(value, (FrozenDict, FrozenList, tuple))


def deep_freeze[T](value: T) -> T:
    """Iteratively freeze nested dict/list/tuple graphs, keeping AbortSignal mutable."""
    memo: dict[int, Any] = {}
    root_holder: list[Any] = [None]
    pending: list[tuple[str, Any, Any, Any]] = []

    def _schedule(node: Any, setter: Callable[[Any], None]) -> None:
        if node is None or isinstance(
            node, (bool, int, float, str, bytes, AbortSignal)
        ):
            setter(node)
            return
        node_id = id(node)
        if node_id in memo:
            setter(memo[node_id])
            return
        if isinstance(node, Mapping):
            frozen_dict = FrozenDict()
            memo[node_id] = frozen_dict
            memo[id(frozen_dict)] = frozen_dict
            setter(frozen_dict)
            for key, item in reversed(list(node.items())):
                pending.append(("dict", frozen_dict, key, item))
            return
        if isinstance(node, list):
            frozen_list = FrozenList([None] * len(node))
            memo[node_id] = frozen_list
            memo[id(frozen_list)] = frozen_list
            setter(frozen_list)
            for idx in range(len(node) - 1, -1, -1):
                pending.append(("list", frozen_list, idx, node[idx]))
            return
        if isinstance(node, tuple):
            buf: list[Any] = [None] * len(node)
            pending.append(("tuple_done", setter, buf, None))
            for idx in range(len(node) - 1, -1, -1):
                pending.append(("list", buf, idx, node[idx]))
            return
        setter(node)

    def _set_root(frozen: Any) -> None:
        root_holder[0] = frozen

    _schedule(value, _set_root)
    while pending:
        kind, target, key_or_idx, item = pending.pop()
        if kind == "dict":
            target_dict = target
            key = key_or_idx

            def _assign_dict(
                frozen: Any, d: dict[Any, Any] = target_dict, k: Any = key
            ) -> None:
                dict.__setitem__(d, k, frozen)

            _schedule(item, _assign_dict)
        elif kind == "list":
            target_list = target
            index = key_or_idx

            def _assign_list(
                frozen: Any, lst: list[Any] = target_list, i: int = index
            ) -> None:
                list.__setitem__(lst, i, frozen)

            _schedule(item, _assign_list)
        else:
            key_or_idx_setter = target
            frozen_tuple = tuple(key_or_idx)
            key_or_idx_setter(frozen_tuple)

    return root_holder[0]  # type: ignore[no-any-return]


def _walk_json(value: Any, *, detach: bool) -> Any:
    """Iteratively validate lossless JSON and optionally produce a detached copy."""
    active_ids: set[int] = set()
    root_holder: list[Any] = [True]
    # Frame: [kind, source, items_or_keys, index, target]
    frames: list[list[Any]] = []
    parent_frame: list[Any] | None = None
    parent_key: Any = None
    current: Any = value

    def _assign(item: Any) -> None:
        if parent_frame is None:
            if detach:
                root_holder[0] = item
        elif parent_frame[0] == "list":
            if parent_frame[4] is not None:
                parent_frame[4].append(item)
        elif parent_frame[4] is not None:
            parent_frame[4][parent_key] = item

    while True:
        if current is None or isinstance(current, (bool, str)):
            _assign(current)
        elif isinstance(current, int):
            if not is_safe_integer(current):
                return _SENTINEL_INVALID
            _assign(current)
        elif isinstance(current, float):
            if not math.isfinite(current) or (
                current == 0.0 and math.copysign(1.0, current) < 0.0
            ):
                return _SENTINEL_INVALID
            _assign(current)
        else:
            obj_id = id(current)
            if obj_id in active_ids:
                return _SENTINEL_INVALID
            if type(current) in (list, FrozenList):
                target_list: list[Any] | None = [] if detach else None
                if target_list is not None:
                    _assign(target_list)
                active_ids.add(obj_id)
                frames.append(["list", current, len(current), 0, target_list])
            elif type(current) in (dict, FrozenDict):
                keys: list[Any] = []
                for key in current:
                    if not isinstance(key, str):
                        return _SENTINEL_INVALID
                    keys.append(key)
                target_dict: dict[str, Any] | None = {} if detach else None
                if target_dict is not None:
                    _assign(target_dict)
                active_ids.add(obj_id)
                frames.append(["dict", current, keys, 0, target_dict])
            else:
                return _SENTINEL_INVALID

        while True:
            if not frames:
                return root_holder[0]
            frame = frames[-1]
            limit = frame[2] if frame[0] == "list" else len(frame[2])
            if frame[3] >= limit:
                active_ids.discard(id(frame[1]))
                frames.pop()
                continue
            idx = frame[3]
            frame[3] = idx + 1
            parent_frame = frame
            if frame[0] == "list":
                current = frame[1][idx]
            else:
                key = frame[2][idx]
                parent_key = key
                current = frame[1][key]
            break


def snapshot_json_value(value: Any) -> Any | None:
    """Validate and detach lossless JSON, returning None when not JSON-lossless."""
    result = _walk_json(value, detach=True)
    return None if result is _SENTINEL_INVALID else result


def is_json_value(value: Any) -> bool:
    """Return True when *value* satisfies the lossless JSON value invariant."""
    return _walk_json(value, detach=False) is not _SENTINEL_INVALID


def deep_equal_json(left: Any, right: Any) -> bool:
    """Iteratively compare two JSON-compatible values structurally."""
    stack: list[tuple[Any, Any]] = [(left, right)]
    while stack:
        a, b = stack.pop()
        if type(a) is bool or type(b) is bool:
            if type(a) is not type(b) or a != b:
                return False
            continue
        if a is b:
            continue
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            for item_a, item_b in zip(a, b, strict=True):
                stack.append((item_a, item_b))
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            if len(a) != len(b):
                return False
            for key, val in a.items():
                if key not in b:
                    return False
                stack.append((val, b[key]))
            continue
        if (
            isinstance(a, (int, float, str, type(None)))
            and isinstance(b, (int, float, str, type(None)))
            and a == b
        ):
            continue
        return False
    return True


assertNever = assert_never
deepFreeze = deep_freeze
snapshotJsonValue = snapshot_json_value
isJsonValue = is_json_value
deepEqualJson = deep_equal_json
