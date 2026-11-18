"""Unit tests for installed-component discovery via entry points."""

from __future__ import annotations

from typing import Any

import pytest

from magitech_harness.components import discovery


class _FakeEP:
    def __init__(self, name: str, provider: Any) -> None:
        self.name = name
        self._provider = provider

    def load(self) -> Any:
        return self._provider


def test_discover_returns_named_dicts_only(monkeypatch: pytest.MonkeyPatch) -> None:
    good = {
        "name": "foo-tool",
        "version": "1.0.0",
        "description": "does foo",
        "target": "foo.component:apply",
    }
    broken = _FakeEP("broken", lambda: (_ for _ in ()).throw(RuntimeError("bad")))
    monkeypatch.setattr(
        discovery,
        "entry_points",
        lambda group: [
            _FakeEP("good", lambda: [good, {"version": "no name"}, 42]),
            broken,
        ],
    )
    assert discovery.discover_installed_components() == [good]


def test_discover_no_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discovery, "entry_points", lambda group: [])
    assert discovery.discover_installed_components() == []
