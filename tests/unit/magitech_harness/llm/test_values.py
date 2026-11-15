"""Unit tests for immutable container freezing and lossless JSON utilities."""

from __future__ import annotations

import copy
import json
import math
import weakref

import pytest

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.llm.values import (
    MAX_SAFE_INTEGER,
    FrozenDict,
    FrozenList,
    assert_never,
    deep_equal_json,
    deep_freeze,
    is_frozen,
    is_json_value,
    is_non_negative_safe_integer,
    is_safe_integer,
    snapshot_json_value,
)


def test_safe_integer_predicates() -> None:
    assert is_safe_integer(0)
    assert is_safe_integer(MAX_SAFE_INTEGER)
    assert is_safe_integer(-MAX_SAFE_INTEGER)
    assert not is_safe_integer(MAX_SAFE_INTEGER + 1)
    assert not is_safe_integer(-MAX_SAFE_INTEGER - 1)
    assert not is_safe_integer(True)
    assert not is_safe_integer(1.5)
    assert is_safe_integer(1.0) is False

    assert is_non_negative_safe_integer(0)
    assert is_non_negative_safe_integer(42)
    assert not is_non_negative_safe_integer(-1)
    assert not is_non_negative_safe_integer(False)


def test_assert_never_raises_with_and_without_context() -> None:
    with pytest.raises(ValueError, match="unreachable variant in test-ctx"):
        assert_never("bad", "test-ctx")
    with pytest.raises(ValueError, match="unreachable variant"):
        assert_never({"kind": "x"})
    with pytest.raises(ValueError, match="unreachable variant"):
        assert_never(object())


def test_frozen_dict_and_list_immutability_json_equality_and_weakref() -> None:
    signal = AbortSignal()
    nested: dict[str, object] = {
        "a": [1, (2, {"b": "ok"})],
        "signal": signal,
        "custom": object(),
    }
    nested["self"] = nested
    frozen = deep_freeze(nested)

    assert is_frozen(frozen)
    assert is_frozen(frozen["a"])
    assert isinstance(frozen["a"][1], tuple)
    assert frozen["signal"] is signal
    assert frozen["self"] is frozen
    assert not is_frozen({"plain": 1})

    # Equality, JSON serialization, copying, and weakrefs all work naturally.
    assert FrozenDict({"k": 1}) == {"k": 1}
    assert FrozenList([1, 2]) == [1, 2]
    assert json.dumps(deep_freeze({"k": [1, 2]})) == '{"k": [1, 2]}'
    assert weakref.ref(deep_freeze({"x": 1}))() == {"x": 1}
    assert weakref.ref(deep_freeze([1, 2]))() == [1, 2]

    plain_dict_copy = copy.deepcopy(deep_freeze({"a": [1, {"b": 2}]}))
    assert type(plain_dict_copy) is dict
    assert type(plain_dict_copy["a"]) is list
    assert type(copy.copy(FrozenDict({"k": 1}))) is dict
    assert type(copy.copy(FrozenList([1]))) is list

    fd = FrozenDict({"k": 1})
    with pytest.raises(TypeError, match="Cannot modify frozen dict"):
        fd["k"] = 2
    with pytest.raises(TypeError, match="Cannot modify frozen dict"):
        del fd["k"]
    with pytest.raises(TypeError, match="Cannot modify frozen dict"):
        fd.pop("k")
    with pytest.raises(TypeError, match="Cannot modify frozen dict"):
        fd.popitem()
    with pytest.raises(TypeError, match="Cannot modify frozen dict"):
        fd.clear()
    with pytest.raises(TypeError, match="Cannot modify frozen dict"):
        fd.update({"m": 2})
    with pytest.raises(TypeError, match="Cannot modify frozen dict"):
        fd.setdefault("m", 2)
    with pytest.raises(TypeError, match="Cannot modify frozen dict"):
        fd |= {"m": 2}

    fl = FrozenList([1, 2])
    with pytest.raises(TypeError, match="Cannot modify frozen list"):
        fl[0] = 9
    with pytest.raises(TypeError, match="Cannot modify frozen list"):
        del fl[0]
    with pytest.raises(TypeError, match="Cannot modify frozen list"):
        fl.append(3)
    with pytest.raises(TypeError, match="Cannot modify frozen list"):
        fl.extend([3])
    with pytest.raises(TypeError, match="Cannot modify frozen list"):
        fl.insert(0, 3)
    with pytest.raises(TypeError, match="Cannot modify frozen list"):
        fl.pop()
    with pytest.raises(TypeError, match="Cannot modify frozen list"):
        fl.remove(1)
    with pytest.raises(TypeError, match="Cannot modify frozen list"):
        fl.clear()
    with pytest.raises(TypeError, match="Cannot modify frozen list"):
        fl.sort()
    with pytest.raises(TypeError, match="Cannot modify frozen list"):
        fl.reverse()
    with pytest.raises(TypeError, match="Cannot modify frozen list"):
        fl += [3]
    with pytest.raises(TypeError, match="Cannot modify frozen list"):
        fl *= 2


def test_json_snapshot_validation_and_deep_equality() -> None:
    src = {"arr": [1, True, None, "x", 2.5], "obj": {"k": 0.0}}
    snap = snapshot_json_value(src)
    assert snap == src
    assert snap is not src
    assert snap["arr"] is not src["arr"]
    assert is_json_value(src)
    assert is_json_value(deep_freeze(src))

    # Reject non-safe ints, non-finite floats, negative zero, non-str keys, cycles, objects.
    cyclic_list: list[object] = []
    cyclic_list.append(cyclic_list)

    class _SubDict(dict[str, int]):
        pass

    assert snapshot_json_value(-0.0) is None
    assert not is_json_value(-0.0)
    assert snapshot_json_value(math.inf) is None
    assert snapshot_json_value(math.nan) is None
    assert snapshot_json_value(MAX_SAFE_INTEGER + 1) is None
    assert snapshot_json_value({1: "non-str-key"}) is None
    assert snapshot_json_value(cyclic_list) is None
    assert snapshot_json_value(object()) is None
    assert snapshot_json_value(_SubDict({"a": 1})) is None
    assert snapshot_json_value((1, 2)) is None

    # Shared non-cyclic DAG references are allowed.
    shared = {"v": 1}
    assert is_json_value([shared, shared])

    # Deep structural equality handles bool vs int and nested differences cleanly.
    assert deep_equal_json({"a": [1, "x"], "b": False}, {"a": [1, "x"], "b": False})
    assert not deep_equal_json(True, 1)
    assert not deep_equal_json(False, 0)
    assert not deep_equal_json([1], [1, 2])
    assert not deep_equal_json([1], [2])
    assert not deep_equal_json({"a": 1}, {"a": 1, "b": 2})
    assert not deep_equal_json({"a": 1}, {"b": 1})
    assert not deep_equal_json({"a": 1}, {"a": 2})
    assert not deep_equal_json("1", 1)
    assert not deep_equal_json(object(), object())
    assert is_json_value(42)
    assert deep_equal_json(100000, 100000.0)
