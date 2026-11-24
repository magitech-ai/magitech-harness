"""Unit tests for the context contributor service."""

from __future__ import annotations

from typing import Any

import pytest
from animas import Context

from magitech_harness.context.service import ContextService


class _Contributor:
    def __init__(self, name: str, text: str | None = "ctx") -> None:
        self._name = name
        self._text = text

    @property
    def name(self) -> str:
        return self._name

    def contribute(self, request: Any) -> str | None:
        return self._text


def test_context_service_registers_and_lists_contributors() -> None:
    ctx = Context()
    svc = ContextService(ctx)
    contributor = _Contributor("time", "now")
    disposer = svc.register(contributor)
    assert svc.list_contributors() == ["time"]
    assert svc.contribute({}) == ["now"]
    disposer()
    assert svc.list_contributors() == []


def test_context_service_rejects_duplicate_and_empty_names() -> None:
    ctx = Context()
    svc = ContextService(ctx)
    svc.register(_Contributor("time", "now"))
    with pytest.raises(ValueError, match="already registered"):
        svc.register(_Contributor("time", "again"))
    with pytest.raises(ValueError, match="non-empty"):
        svc.register(_Contributor("   ", "x"))


def test_context_service_skips_failing_and_empty_contributors() -> None:
    ctx = Context()
    svc = ContextService(ctx)

    class _Failing:
        @property
        def name(self) -> str:
            return "failing"

        def contribute(self, request: Any) -> str | None:
            raise RuntimeError("boom")

    class _Empty:
        @property
        def name(self) -> str:
            return "empty"

        def contribute(self, request: Any) -> str | None:
            return "   "

    svc.register(_Failing())
    svc.register(_Empty())
    assert svc.contribute({}) == []


def test_context_service_mounts_in_the_default_bundle() -> None:
    from magitech_harness import create_harness

    ctx = create_harness()
    assert ctx.get("context", None, strict=False) is not None
