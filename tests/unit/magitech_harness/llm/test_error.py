"""Unit tests for LLM taxonomy errors, predicates, and error_chain formatting."""

from __future__ import annotations

from typing import Any

from magitech_harness.llm.error import (
    ACCOUNT_QUOTA_EXCEEDED_CODE,
    CONTEXT_WINDOW_EXCEEDED_CODE,
    EMPTY_RESPONSE_CODE,
    IMAGE_OFFLOAD_REQUIRED_CODE,
    INVALID_CREDENTIAL_CODE,
    QUOTA_EXCEEDED_CODE,
    HarnessError,
    RemoteError,
    error_chain,
    is_context_window_exceeded_error,
    is_harness_error,
    is_quota_exceeded_error,
)


def test_error_constants_and_classes() -> None:
    assert CONTEXT_WINDOW_EXCEEDED_CODE == "CONTEXT_WINDOW_EXCEEDED"
    assert QUOTA_EXCEEDED_CODE == "QUOTA"
    assert ACCOUNT_QUOTA_EXCEEDED_CODE == "ACCOUNT_QUOTA"
    assert EMPTY_RESPONSE_CODE == "EMPTY_RESPONSE"
    assert INVALID_CREDENTIAL_CODE == "INVALID_CREDENTIAL"
    assert IMAGE_OFFLOAD_REQUIRED_CODE == "IMAGE_OFFLOAD_REQUIRED"

    root = ValueError("socket hang up")
    err = HarnessError("call failed", "TRANSPORT", cause=root)
    assert err.code == "TRANSPORT"
    assert err.message == "call failed"
    assert err.__cause__ is root
    assert is_harness_error(err)
    assert not is_harness_error(root)

    # Non-exception cause preserves .cause without chaining __cause__.
    err_non_exc = HarnessError("call failed", "SERVER", cause="raw detail")
    assert err_non_exc.cause == "raw detail"
    assert err_non_exc.__cause__ is None

    remote = RemoteError("rpc/bad", "rejected", {"id": "1"}, cause=root)
    assert remote.is_mth_remote_error is True
    assert remote.code == "rpc/bad"
    assert remote.details == {"id": "1"}
    assert remote.__cause__ is root

    remote_raw = RemoteError("rpc/raw", "raw", {"id": "2"}, cause="info")
    assert remote_raw.__cause__ is None


def test_context_window_and_quota_predicates() -> None:
    assert is_context_window_exceeded_error("maximum context length is 128000 tokens")
    assert is_context_window_exceeded_error("input is too long for this model")
    assert is_context_window_exceeded_error("request too large for model context")
    assert is_context_window_exceeded_error(
        "input exceeds the model context window limit"
    )
    assert not is_context_window_exceeded_error("rate limit exceeded")

    assert is_quota_exceeded_error("exceeded your current quota")
    assert is_quota_exceeded_error("insufficient_quota for project")
    assert is_quota_exceeded_error("usage-limit-exceeded")
    assert is_quota_exceeded_error("out of credits")
    assert not is_quota_exceeded_error("too many requests per second")


def test_error_chain_formatting_cycles_diamonds_aggregates_and_hostiles() -> None:
    inner = RuntimeError("inner failure")
    outer = RuntimeError("outer failure")
    outer.__cause__ = inner
    assert error_chain(outer) == "outer failure: inner failure"

    # Identical cause collapses cleanly.
    same = RuntimeError("same")
    dup = RuntimeError("same")
    dup.__cause__ = same
    assert error_chain(dup) == "same"

    # Active cycle produces <circular cause>, diamond DAG renders both branches.
    cyc_a: Any = RuntimeError("node-a")
    cyc_b: Any = RuntimeError("node-b")
    cyc_a.cause = cyc_b
    cyc_b.cause = cyc_a
    assert error_chain(cyc_a) == "node-a: node-b: <circular cause>"

    shared = RuntimeError("shared-leaf")
    branch_left: Any = RuntimeError("left")
    branch_left.cause = shared
    branch_right: Any = RuntimeError("right")
    branch_right.cause = shared
    agg = ExceptionGroup("multi", [branch_left, branch_right])
    assert error_chain(agg) == "multi [left: shared-leaf; right: shared-leaf]"

    # Empty exception falls back to type name; empty aggregate falls back safely.
    class _CustomAgg(Exception):
        def __init__(self) -> None:
            super().__init__("wrapper")
            self.errors: list[Any] = []

    assert error_chain(ValueError()) == "ValueError"
    assert error_chain(_CustomAgg()) == "wrapper"

    # Dict with message vs non-str message vs plain non-exception vs own-dict str message.
    assert error_chain({"message": "plain obj msg"}) == "plain obj msg"
    assert error_chain({"message": 123}) == "{'message': 123}"
    assert error_chain(404) == "404"
    own_msg_err: Any = RuntimeError("outer")
    own_msg_err.message = "own dict msg"
    assert error_chain(own_msg_err) == "own dict msg"

    class _Hostile:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    class _HostileMsgExc(Exception):
        def __str__(self) -> str:
            raise RuntimeError("bad str")

    assert error_chain(_Hostile()) == "<unrenderable value>"
    assert error_chain(_HostileMsgExc()) == "<unrenderable value>"

    class _ThrowingDictError(Exception):
        @property
        def __dict__(self) -> dict[str, Any]:
            raise RuntimeError("hostile dict")

    assert error_chain(_ThrowingDictError("dict boom")) == "dict boom"
