"""Unit tests for AbortSignal cooperative cancellation."""

import pytest

from magitech_harness.agent.cancellation import AbortError, AbortSignal


def test_abort_signal_initial_state() -> None:
    signal = AbortSignal()
    assert not signal.is_aborted
    assert not signal.aborted
    assert signal.reason is None
    # throwIfAborted should not raise
    signal.throw_if_aborted()


def test_abort_signal_abort_with_reason_and_add_listener() -> None:
    signal = AbortSignal()
    events = []
    disposer = signal.add_listener(lambda: events.append("aborted"))
    signal.abort("Timeout")
    assert signal.is_aborted
    assert signal.aborted
    assert signal.reason == "Timeout"
    assert events == ["aborted"]
    with pytest.raises(AbortError, match="Operation was aborted: Timeout"):
        signal.throw_if_aborted()
    disposer()


def test_abort_signal_abort_triggers_listeners() -> None:
    signal = AbortSignal()
    called = []

    def on_abort() -> None:
        called.append(True)

    disposer = signal.on_abort(on_abort)
    assert not called

    signal.abort()
    assert signal.is_aborted
    assert called == [True]

    with pytest.raises(AbortError, match="Operation was aborted"):
        signal.throw_if_aborted()

    # Second abort call is a no-op
    signal.abort()
    assert called == [True]

    # Calling disposer after abort
    disposer()


def test_abort_signal_on_abort_when_already_aborted() -> None:
    signal = AbortSignal()
    signal.abort()

    called = []
    disposer = signal.on_abort(lambda: called.append(True))
    assert called == [True]
    disposer()


def test_abort_signal_dispose_before_abort() -> None:
    signal = AbortSignal()
    called = []

    disposer = signal.on_abort(lambda: called.append(True))
    disposer()

    signal.abort()
    assert not called
