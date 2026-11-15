"""Unit tests for AbortSignal cooperative cancellation."""

import asyncio

import pytest

from magitech_harness.agent.cancellation import (
    MAX_TIMER_DELAY_MS,
    AbortController,
    AbortError,
    AbortSignal,
    TimeoutReason,
    clamp_timeout,
    combine_signals,
    deadline,
    timeout_of,
)


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


@pytest.mark.asyncio
async def test_timeout_reason_clamp_and_deadline() -> None:
    """Verifies TimeoutReason, clamp_timeout, timeout_of, and deadline fusion."""

    reason = TimeoutReason("TOOL_TIMEOUT", 250)
    assert reason.code == "TOOL_TIMEOUT"
    assert reason.timeout_ms == 250
    assert "TOOL_TIMEOUT after 250ms" in str(reason)

    assert clamp_timeout(None, 5000, 10000) == 5000
    assert clamp_timeout(20000, 5000, 10000) == 10000
    assert clamp_timeout(1200, 5000, 10000) == 1200
    with pytest.raises(ValueError, match="positive finite"):
        clamp_timeout(0, 5000, 10000, "customTimeoutMs")
    with pytest.raises(ValueError, match="positive finite"):
        clamp_timeout(float("nan"), 5000, 10000)

    # No-timer sentinel (timeout_ms <= 0)
    upstream = AbortController()
    no_timer = deadline(upstream.signal, 0, "NOOP")
    assert no_timer.signal is upstream.signal
    no_timer.dispose()
    assert deadline(None, -1, "NOOP").signal.aborted is False

    # Invalid delays raise ValueError
    with pytest.raises(ValueError, match="positive finite"):
        deadline(None, float("inf"), "BAD")
    with pytest.raises(ValueError, match="no greater than"):
        deadline(None, MAX_TIMER_DELAY_MS + 1, "BAD")

    # Context-manager & upstream cancellation wins without TimeoutReason
    upstream2 = AbortController()
    with deadline(upstream2.signal, 500, "OP_TIMEOUT") as dl:
        upstream2.abort("user-stop")
        assert dl.signal.aborted is True
        assert timeout_of(dl.signal) is None

    # Already aborted upstream fuses immediately
    upstream3 = AbortController()
    upstream3.abort("already")
    dl_already = deadline(upstream3.signal, 500, "OP_TIMEOUT")
    assert dl_already.signal.aborted is True
    assert dl_already.signal.reason == "already"
    dl_already.dispose()

    # Timer expires and stamps TimeoutReason
    dl_expire = deadline(None, 10, "MODEL_LISTING_TIMEOUT")
    await asyncio.sleep(0.05)
    assert dl_expire.signal.aborted is True
    matched = timeout_of(dl_expire.signal, "MODEL_LISTING_TIMEOUT")
    assert matched is not None
    assert matched.timeout_ms == 10
    assert timeout_of(dl_expire.signal, "OTHER_CODE") is None
    assert timeout_of({"reason": TimeoutReason("A", 1)}, "A") is None
    assert timeout_of(dl_expire.signal) is matched
    dl_expire.dispose()
    dl_expire.dispose()  # Idempotent second dispose

    # Already aborted upstream still handles dispose cleanly
    with pytest.raises(ValueError, match="positive finite"):
        deadline(None, True, "BAD")  # type: ignore[arg-type]


def test_combine_signals_fuses_preaborted_live_and_dispose() -> None:
    """Fuses signals, forwards first abort reason, and cleans up on dispose."""
    s1 = AbortController()
    s2 = AbortController()
    fused = combine_signals(s1.signal, s2.signal)
    assert fused.signal.aborted is False
    s2.abort("second")
    assert fused.signal.aborted is True
    assert fused.signal.reason == "second"
    fused.dispose()

    # Pre-aborted upstream aborts immediately and dispose is no-op
    fused_pre = combine_signals(s2.signal, s1.signal)
    assert fused_pre.signal.aborted is True
    assert fused_pre.signal.reason == "second"
    fused_pre.dispose()

    # Disposing un-aborted fusion detaches listeners cleanly
    a = AbortController()
    b = AbortController()
    fused_live = combine_signals(a.signal, b.signal)
    fused_live.dispose()
    a.abort("later")
    assert fused_live.signal.aborted is False
