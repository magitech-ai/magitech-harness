"""Unit tests for the log-backed session title service."""

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from anima import Context, EventMode

from magitech_harness import MockLLMProvider, create_harness
from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.llm.service import LLMService
from magitech_harness.session.events import EventType, SessionEvent
from magitech_harness.session.projection_registry import SessionProjectionRegistry
from magitech_harness.session.projections import (
    TITLE_INPUT_PROJECTION,
    TITLE_PROJECTION,
)
from magitech_harness.session.service import SessionService
from magitech_harness.session.store import Session, SessionStore
from magitech_harness.session.title import (
    SessionTitleInvalidError,
    SessionTitleService,
    clean_title_text,
    collect_title_messages,
    current_route,
    eligible_title_message,
    fallback_session_title,
    fold_session_title,
    normalize_session_title,
    resolve_title_config,
    title_snapshot_from_event,
    truncate_title_utf8,
    validate_title_result,
)


def _service(
    config: dict[str, Any] | None = None,
) -> tuple[Context, SessionService]:
    ctx = Context()
    return ctx, SessionService(ctx, config or {})


def _user(
    session: Session,
    service: SessionService,
    text: str,
    *,
    source_kind: str = "user",
) -> SessionEvent:
    return service.log_event(
        session,
        EventType.USER_MESSAGE,
        {
            "content": [{"type": "text", "text": text}],
            "source": {"kind": source_kind},
        },
    )


def _header(
    session: Session,
    service: SessionService,
    *,
    provider: str = "prov",
    model: str = "mod",
) -> SessionEvent:
    return service.log_event(
        session,
        EventType.REQUEST_HEADER,
        {
            "header": {"config": {"provider": provider, "model": model}},
            "reason": "initial",
        },
    )


def _title_events(session: Session) -> list[SessionEvent]:
    return [event for event in session.events if event.type == "session/title"]


async def _pump() -> None:
    """Let deferred title tasks scheduled on the running loop take one step."""
    await asyncio.sleep(0)


class _FakeProvider:
    """Scriptable title provider recording every request."""

    def __init__(
        self,
        *,
        title: str = "Provider Title",
        automatic: str = "first-prompt",
        result: dict[str, Any] | None = None,
        fail: BaseException | None = None,
    ) -> None:
        self.id = "fake-provider"
        self.automatic = automatic
        self.requests: list[Any] = []
        self._title = title
        self._result = result
        self._fail = fail
        self.gate: asyncio.Event | None = None

    async def generate(self, request: Any) -> dict[str, Any]:
        self.requests.append(request)
        if self.gate is not None and len(self.requests) == 1:
            await self.gate.wait()
        if self._fail is not None:
            raise self._fail
        if self._result is not None:
            return self._result
        return {
            "title": self._title,
            "messageSeqs": [message["seq"] for message in request.messages],
        }


class _RecordingProvider:
    """Provider that records generate calls without a title service."""

    def __init__(self) -> None:
        self.id = "recording-provider"
        self.automatic = "first-prompt"
        self.requests: list[Any] = []

    async def generate(self, request: Any) -> dict[str, Any]:
        self.requests.append(request)
        return {"title": "Recorded", "messageSeqs": [request.messages[0]["seq"]]}


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------


def test_resolve_title_config_valid() -> None:
    config = resolve_title_config(
        {
            "fallback_max_words": 5,
            "fallback_max_bytes": 40,
            "max_title_bytes": 80,
        }
    )
    assert config.fallback_max_words == 5
    assert config.fallback_max_bytes == 40
    assert config.max_title_bytes == 80


@pytest.mark.parametrize(
    "config",
    [
        None,
        {},
        {"fallback_max_words": 5},
        {"fallback_max_words": 5, "fallback_max_bytes": 40},
    ],
)
def test_resolve_title_config_missing(config: Any) -> None:
    with pytest.raises(ValueError, match="session-title:"):
        resolve_title_config(config)


@pytest.mark.parametrize(
    "name,value",
    [
        ("fallback_max_words", 0),
        ("fallback_max_words", -1),
        ("fallback_max_words", 1.5),
        ("fallback_max_words", True),
        ("fallback_max_words", "5"),
        ("fallback_max_bytes", 0),
        ("max_title_bytes", 0),
    ],
)
def test_resolve_title_config_non_positive(name: str, value: Any) -> None:
    config = {
        "fallback_max_words": 5,
        "fallback_max_bytes": 40,
        "max_title_bytes": 80,
        name: value,
    }
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        resolve_title_config(config)


def test_resolve_title_config_fallback_exceeds_max() -> None:
    with pytest.raises(
        ValueError, match="fallback_max_bytes must not exceed max_title_bytes"
    ):
        resolve_title_config(
            {
                "fallback_max_words": 5,
                "fallback_max_bytes": 81,
                "max_title_bytes": 80,
            }
        )


def test_service_config_defaults_rename_cap() -> None:
    _, service = _service()
    session = service.create("s1")
    # The shipped 80-byte default caps a rename.
    snapshot = service.title_service.rename(session, "x" * 100)
    assert snapshot["title"] == "x" * 80


async def test_service_config_defaults_fallback_words() -> None:
    _, service = _service()
    session = service.create("s1")
    _user(
        session,
        service,
        "one two three four five six seven eight nine ten",
    )
    await _pump()
    assert service.title_service.get(session)["title"] == "one two three four five"


async def test_service_config_overrides_limits() -> None:
    _, service = _service(
        {
            "title_fallback_max_words": 2,
            "title_fallback_max_bytes": 10,
            "title_max_title_bytes": 20,
        }
    )
    session = service.create("s1")
    _user(session, service, "one two three four five six")
    await _pump()
    assert service.title_service.get(session)["title"] == "one two"
    snapshot = service.title_service.rename(session, "a" * 30)
    assert snapshot["title"] == "a" * 20


def test_service_config_invalid_raises_at_construction() -> None:
    ctx = Context()
    with pytest.raises(
        ValueError, match="fallback_max_bytes must not exceed max_title_bytes"
    ):
        SessionService(
            ctx,
            {"title_fallback_max_bytes": 999, "title_max_title_bytes": 10},
        )


def test_title_service_direct_construction_requires_config() -> None:
    ctx = Context()
    registry = SessionProjectionRegistry(ctx)
    title_handle = registry.register(TITLE_PROJECTION)
    title_input_handle = registry.register(TITLE_INPUT_PROJECTION)
    with pytest.raises(ValueError, match="configuration is required"):
        SessionTitleService(
            ctx, SessionStore(ctx), title_handle, title_input_handle, None
        )


# ---------------------------------------------------------------------------
# Normalization.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  hello   world  ", "hello world"),
        ("\x1b[31mRed\x1b[0m title", "Red title"),
        ("\x1b]0;osc title\x07rest", "rest"),
        ("a\x00b\x07c\x7fd", "abcd"),
        ("a\u200bb\u200ec", "abc"),
        ("a\ufeffb", "ab"),
        ("one\ntwo\tthree", "one two three"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_clean_title_text(raw: str, expected: str) -> None:
    assert clean_title_text(raw) == expected


@pytest.mark.parametrize(
    "text,max_bytes,expected",
    [
        ("abcd", 4, "abcd"),
        ("abcde", 4, "abcd"),
        ("aé", 2, "a"),
        ("aé", 3, "aé"),
        ("a😀b", 4, "a"),
        ("a😀b", 5, "a😀"),
        ("a😀b", 8, "a😀b"),
    ],
)
def test_truncate_title_utf8(text: str, max_bytes: int, expected: str) -> None:
    assert truncate_title_utf8(text, max_bytes) == expected


@pytest.mark.parametrize("max_bytes", [0, -1, True, "3"])
def test_truncate_title_utf8_invalid_budget(max_bytes: Any) -> None:
    with pytest.raises(ValueError, match="max_bytes must be a positive integer"):
        truncate_title_utf8("abc", max_bytes)


def test_normalize_session_title_strips_and_truncates() -> None:
    assert (
        normalize_session_title("  \x1b[1mHello   World\x1b[0m  ", 80) == "Hello World"
    )
    assert normalize_session_title("ab cd", 3) == "ab"
    assert normalize_session_title("   ", 80) == ""


@pytest.mark.parametrize(
    "text,max_words,max_bytes,expected",
    [
        ("one two three four five six", 5, 80, "one two three four five"),
        ("one two three", 5, 7, "one two"),
        ("one two three", 5, 3, "one"),
        ("  spaced   out  words ", 2, 80, "spaced out"),
        ("", 5, 40, ""),
        ("   ", 5, 40, ""),
    ],
)
def test_fallback_session_title(
    text: str, max_words: int, max_bytes: int, expected: str
) -> None:
    assert fallback_session_title(text, max_words, max_bytes) == expected


@pytest.mark.parametrize("max_words", [0, -1, True, "5"])
def test_fallback_session_title_invalid_words(max_words: Any) -> None:
    with pytest.raises(ValueError, match="max_words must be a positive integer"):
        fallback_session_title("hello world", max_words, 40)


# ---------------------------------------------------------------------------
# Eligibility and folds.
# ---------------------------------------------------------------------------


def _raw_event(event_type: str, payload: dict[str, Any], seq: int) -> SessionEvent:
    return SessionEvent(
        id=f"s1:{seq}",
        session_id="s1",
        type=event_type,
        timestamp=1700000000.0 + seq,
        payload=payload,
        seq=seq,
    )


def test_eligible_title_message() -> None:
    event = _raw_event(
        "user/message",
        {
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "url": "x"},
                {"type": "text", "text": "world"},
            ],
            "source": {"kind": "user"},
        },
        3,
    )
    assert eligible_title_message(event) == {"seq": 3, "text": "hello\nworld"}


@pytest.mark.parametrize(
    "event_type,payload",
    [
        (
            "assistant/message",
            {"content": [{"type": "text", "text": "hi"}], "source": {"kind": "user"}},
        ),
        (
            "user/message",
            {"content": [{"type": "text", "text": "hi"}], "source": {"kind": "system"}},
        ),
        ("user/message", {"content": [{"type": "text", "text": "hi"}]}),
        ("user/message", {"source": {"kind": "user"}}),
        (
            "user/message",
            {"content": [{"type": "image", "url": "x"}], "source": {"kind": "user"}},
        ),
        (
            "user/message",
            {"content": [{"type": "text", "text": "   "}], "source": {"kind": "user"}},
        ),
        (
            "user/message",
            {
                "content": [{"type": "text", "text": "\x1b[31m\x1b[0m"}],
                "source": {"kind": "user"},
            },
        ),
    ],
)
def test_eligible_title_message_rejects(
    event_type: str, payload: dict[str, Any]
) -> None:
    assert eligible_title_message(_raw_event(event_type, payload, 0)) is None


def test_fold_session_title_latest_wins() -> None:
    events = [
        _raw_event(
            "session/title",
            {"title": "First", "messageSeqs": [0], "source": {"kind": "fallback"}},
            1,
        ),
        _raw_event("turn/end", {"turn": 1}, 2),
        _raw_event(
            "session/title",
            {"title": "Second", "messageSeqs": [0], "source": {"kind": "user"}},
            3,
        ),
    ]
    snapshot = fold_session_title(events)
    assert snapshot is not None
    assert snapshot["title"] == "Second"
    assert snapshot["source"] == {"kind": "user"}
    assert snapshot["eventSeq"] == 3
    assert snapshot["updatedAt"] == 1700000003000
    assert snapshot["messageSeqs"] == [0]


def test_fold_session_title_empty_log() -> None:
    assert fold_session_title([]) is None


def test_fold_session_title_skips_non_string_title() -> None:
    events = [
        _raw_event(
            "session/title",
            {"title": "Good", "messageSeqs": [0], "source": {"kind": "fallback"}},
            1,
        ),
        _raw_event(
            "session/title",
            {"title": 42, "messageSeqs": [0], "source": {"kind": "fallback"}},
            2,
        ),
    ]
    assert fold_session_title(events)["title"] == "Good"


def test_title_snapshot_from_event_degrades_malformed_fields() -> None:
    event = _raw_event(
        "session/title", {"title": "T", "messageSeqs": "nope", "source": "nope"}, 4
    )
    snapshot = title_snapshot_from_event(event)
    assert snapshot is not None
    assert snapshot["messageSeqs"] == []
    assert snapshot["source"] == {}
    assert title_snapshot_from_event(_raw_event("turn/end", {}, 5)) is None


def test_collect_title_messages_bounds_at_through_seq() -> None:
    events = [
        _raw_event(
            "user/message",
            {"content": [{"type": "text", "text": "one"}], "source": {"kind": "user"}},
            0,
        ),
        _raw_event(
            "user/message",
            {"content": [{"type": "text", "text": "two"}], "source": {"kind": "user"}},
            1,
        ),
        _raw_event(
            "user/message",
            {
                "content": [{"type": "text", "text": "three"}],
                "source": {"kind": "user"},
            },
            2,
        ),
    ]
    assert [m["seq"] for m in collect_title_messages(events)] == [0, 1, 2]
    assert [m["seq"] for m in collect_title_messages(events, 1)] == [0, 1]
    assert collect_title_messages(events, -1) == []


def test_current_route() -> None:
    _, service = _service()
    session = service.create("s1")
    assert current_route(session) is None
    service.log_event(
        session,
        EventType.REQUEST_HEADER,
        {"header": {"config": {}}, "reason": "initial"},
    )
    assert current_route(session) is None
    # A header record without a config mapping carries no route either.
    service.log_event(
        session,
        EventType.REQUEST_HEADER,
        {"header": {}, "reason": "initial"},
    )
    assert current_route(session) is None
    _header(session, service, provider="prov", model="mod")
    assert current_route(session) == {"provider": "prov", "model": "mod"}
    _header(session, service, provider="other", model="v2")
    assert current_route(session) == {"provider": "other", "model": "v2"}


# ---------------------------------------------------------------------------
# get / rename.
# ---------------------------------------------------------------------------


def test_get_before_any_title_is_none() -> None:
    _, service = _service()
    session = service.create("s1")
    assert service.title_service.get(session) is None


def test_rename_normalizes_and_pins() -> None:
    _, service = _service()
    session = service.create("s1")
    snapshot = service.title_service.rename(session, "  \x1b[1mMy  Title\x1b[0m ")
    assert snapshot == {
        "title": "My Title",
        "messageSeqs": [],
        "source": {"kind": "user"},
        "eventSeq": 0,
        "updatedAt": snapshot["updatedAt"],
    }
    assert service.title_service.get(session)["title"] == "My Title"
    event = _title_events(session)[0]
    assert event.payload["source"] == {"kind": "user"}
    assert event.payload["messageSeqs"] == []


@pytest.mark.parametrize("title", ["", "   ", "\x1b[31m\x1b[0m", "\x00\x07"])
def test_rename_empty_after_normalization(title: str) -> None:
    _, service = _service()
    session = service.create("s1")
    with pytest.raises(SessionTitleInvalidError) as excinfo:
        service.title_service.rename(session, title)
    assert excinfo.value.name == "SessionTitleInvalidError"
    assert "visible characters" in str(excinfo.value)
    assert _title_events(session) == []


def test_rename_non_live_session_raises_plain_error() -> None:
    _, service = _service()
    detached = Session.create("detached")
    with pytest.raises(ValueError, match="is not live in this store"):
        service.title_service.rename(detached, "Title")


def test_rename_disposed_service_raises() -> None:
    _, service = _service()
    session = service.create("s1")
    service.title_service.dispose()
    with pytest.raises(RuntimeError, match="session-title service disposed"):
        service.title_service.rename(session, "Title")


def test_rename_truncates_to_max_title_bytes() -> None:
    _, service = _service(
        {
            "title_fallback_max_words": 5,
            "title_fallback_max_bytes": 5,
            "title_max_title_bytes": 5,
        }
    )
    session = service.create("s1")
    snapshot = service.title_service.rename(session, "Hello world")
    assert snapshot["title"] == "Hello"


# ---------------------------------------------------------------------------
# refresh.
# ---------------------------------------------------------------------------


async def test_refresh_without_provider_or_messages_returns_none() -> None:
    _, service = _service()
    session = service.create("s1")
    assert await service.title_service.refresh(session) is None


async def test_refresh_without_provider_materializes_fallback() -> None:
    _, service = _service()
    session = service.create("s1")
    _user(session, service, "How do I reset my password for the dashboard?")
    await _pump()
    assert await service.title_service.refresh(session) is not None
    snapshot = service.title_service.get(session)
    assert snapshot["title"] == "How do I reset my"
    assert snapshot["source"] == {"kind": "fallback"}
    assert snapshot["messageSeqs"] == [0]


async def test_refresh_unpins_user_title_without_provider() -> None:
    _, service = _service()
    session = service.create("s1")
    _user(session, service, "How do I reset my password for the dashboard?")
    await _pump()
    service.title_service.rename(session, "My Custom Name")
    assert service.title_service.get(session)["source"] == {"kind": "user"}
    snapshot = await service.title_service.refresh(session)
    assert snapshot is not None
    assert snapshot["title"] == "How do I reset my"
    assert snapshot["source"] == {"kind": "fallback"}
    assert len(_title_events(session)) == 3  # fallback + rename + fallback


async def test_refresh_non_live_session_raises() -> None:
    _, service = _service()
    detached = Session.create("detached")
    with pytest.raises(ValueError, match="is not live in this store"):
        await service.title_service.refresh(detached)


async def test_refresh_disposed_service_raises() -> None:
    _, service = _service()
    session = service.create("s1")
    service.title_service.dispose()
    with pytest.raises(RuntimeError, match="session-title service disposed"):
        await service.title_service.refresh(session)


async def test_refresh_aborted_signal_raises() -> None:
    _, service = _service()
    session = service.create("s1")
    signal = AbortSignal()
    signal.abort("cancelled")
    with pytest.raises(AbortError):
        await service.title_service.refresh(session, signal)


async def test_refresh_with_live_signal_completes() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider()
    service.title_service.register(provider)
    _user(session, service, "Explain quantum computing simply")
    snapshot = await service.title_service.refresh(session, signal=AbortSignal())
    assert snapshot is not None
    assert snapshot["title"] == "Provider Title"


async def test_refresh_signal_aborted_during_user_branch() -> None:
    ctx, service = _service()
    session = service.create("s1")
    _user(session, service, "How do I reset my password for the dashboard?")
    service.title_service.rename(session, "My Renamed Session")
    signal = AbortSignal()

    def abort_on_fallback(session_arg: Session, event: SessionEvent) -> None:
        if (
            event.type == "session/title"
            and event.payload.get("source", {}).get("kind") == "fallback"
        ):
            signal.abort("cancelled during refresh")

    ctx.on("session/event", abort_on_fallback)
    with pytest.raises(AbortError):
        await service.title_service.refresh(session, signal)
    # The fallback landed before the abort was observed.
    assert service.title_service.get(session)["source"] == {"kind": "fallback"}
    await service.title_service.drain()


async def test_refresh_signal_aborted_during_fallback_branch() -> None:
    ctx, service = _service()
    session = service.create("s1")
    _user(session, service, "How do I reset my password for the dashboard?")
    signal = AbortSignal()

    def abort_on_fallback(session_arg: Session, event: SessionEvent) -> None:
        if (
            event.type == "session/title"
            and event.payload.get("source", {}).get("kind") == "fallback"
        ):
            signal.abort("cancelled during refresh")

    ctx.on("session/event", abort_on_fallback)
    with pytest.raises(AbortError):
        await service.title_service.refresh(session, signal)
    assert service.title_service.get(session)["source"] == {"kind": "fallback"}
    await service.title_service.drain()


async def test_refresh_with_provider_uses_current_route() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider()
    service.title_service.register(provider)
    _user(session, service, "Explain quantum computing simply")
    _header(session, service, provider="prov", model="mod")
    await _pump()
    snapshot = await service.title_service.refresh(session)
    assert snapshot is not None
    assert snapshot["title"] == "Provider Title"
    assert snapshot["source"] == {"kind": "provider", "provider": "fake-provider"}
    assert len(provider.requests) == 2  # automatic run + explicit refresh
    assert provider.requests[-1].route == {"provider": "prov", "model": "mod"}
    assert provider.requests[-1].current_title is not None


async def test_refresh_provider_failure_propagates_and_fallback_stands() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider(fail=RuntimeError("model down"))
    service.title_service.register(provider)
    _user(session, service, "How do I reset my password for the dashboard?")
    await _pump()
    # No header arrived, so no automatic run started; the fallback stands.
    assert service.title_service.get(session)["source"] == {"kind": "fallback"}
    with pytest.raises(RuntimeError, match="model down"):
        await service.title_service.refresh(session)
    assert service.title_service.get(session)["title"] == "How do I reset my"


async def test_refresh_supersedes_older_generation() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider(title="Second Title")
    provider.gate = asyncio.Event()
    service.title_service.register(provider)
    _user(session, service, "Explain quantum computing simply")
    first = asyncio.create_task(service.title_service.refresh(session))
    await _pump()
    assert len(provider.requests) == 1
    second = await service.title_service.refresh(session)
    assert second is not None
    assert second["title"] == "Second Title"
    provider.gate.set()
    with pytest.raises(AbortError):
        await first
    assert service.title_service.get(session)["title"] == "Second Title"
    assert len(_title_events(session)) == 2  # fallback + second provider title


# ---------------------------------------------------------------------------
# register.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider",
    [
        None,
        {"id": "x", "automatic": "first-prompt"},
    ],
)
def test_register_invalid_provider(provider: Any) -> None:
    _, service = _service()
    with pytest.raises(ValueError):
        service.title_service.register(provider)


def test_register_provider_with_empty_id() -> None:
    _, service = _service()
    provider = _RecordingProvider()
    provider.id = ""
    with pytest.raises(ValueError, match="non-empty string"):
        service.title_service.register(provider)


def test_register_invalid_automatic_mode() -> None:
    _, service = _service()
    provider = _RecordingProvider()
    provider.automatic = "sometimes"
    with pytest.raises(ValueError, match="automatic mode is invalid"):
        service.title_service.register(provider)


def test_register_provider_without_generate() -> None:
    _, service = _service()

    class _NoGenerate:
        id = "no-generate"
        automatic = "first-prompt"

    with pytest.raises(TypeError, match="requires generate"):
        service.title_service.register(_NoGenerate())


def test_register_second_live_provider_throws() -> None:
    _, service = _service()
    service.title_service.register(_FakeProvider())
    other = _FakeProvider()
    other.id = "other-provider"
    with pytest.raises(ValueError, match="is already registered"):
        service.title_service.register(other)


def test_register_after_dispose_allows_replacement() -> None:
    _, service = _service()
    dispose = service.title_service.register(_FakeProvider())
    dispose()
    other = _FakeProvider(title="Replacement Title")
    other.id = "other-provider"
    service.title_service.register(other)
    # The replacement is live: an explicit refresh reaches it.
    session = service.create("s1")
    _user(session, service, "Explain quantum computing simply")
    snapshot = asyncio.run(service.title_service.refresh(session))
    assert snapshot is not None
    assert snapshot["title"] == "Replacement Title"


def test_register_after_service_dispose_raises() -> None:
    _, service = _service()
    service.title_service.dispose()
    with pytest.raises(RuntimeError, match="session-title service disposed"):
        service.title_service.register(_FakeProvider())


def test_register_disposer_is_idempotent() -> None:
    _, service = _service()
    dispose = service.title_service.register(_FakeProvider())
    dispose()
    dispose()
    # Disposing twice leaves the service usable for a fresh registration.
    service.title_service.register(_FakeProvider())


# ---------------------------------------------------------------------------
# Automatic scheduling: fallback first, then provider.
# ---------------------------------------------------------------------------


async def test_first_prompt_schedules_on_header_after_first_message() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider()
    service.title_service.register(provider)
    _user(session, service, "Explain quantum computing simply")
    await _pump()
    assert len(provider.requests) == 0
    assert service.title_service.get(session)["source"] == {"kind": "fallback"}
    _header(session, service, provider="prov", model="mod")
    await service.title_service.drain()
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert [m["seq"] for m in request.messages] == [0]
    assert request.route == {"provider": "prov", "model": "mod"}
    snapshot = service.title_service.get(session)
    assert snapshot["title"] == "Provider Title"
    assert snapshot["source"] == {"kind": "provider", "provider": "fake-provider"}
    assert snapshot["messageSeqs"] == [0]
    events = _title_events(session)
    assert events[0].payload["source"] == {"kind": "fallback"}
    assert events[1].payload["source"]["kind"] == "provider"


async def test_first_prompt_second_message_does_not_reschedule() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider()
    service.title_service.register(provider)
    _user(session, service, "Explain quantum computing simply")
    _header(session, service, provider="prov", model="mod")
    await service.title_service.drain()
    assert len(provider.requests) == 1
    _user(session, service, "And now explain relativity")
    _header(session, service, provider="prov", model="mod")
    await service.title_service.drain()
    assert len(provider.requests) == 1
    assert service.title_service.get(session)["title"] == "Provider Title"


async def test_all_prompts_schedules_on_every_message() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider(automatic="all-prompts")
    service.title_service.register(provider)
    _user(session, service, "Explain quantum computing simply")
    _header(session, service, provider="prov", model="mod")
    await service.title_service.drain()
    assert len(provider.requests) == 1
    _user(session, service, "And now explain relativity")
    _header(session, service, provider="prov", model="mod")
    await service.title_service.drain()
    assert len(provider.requests) == 2
    # The second run collects both eligible messages (seqs 0 and 4;
    # the fallback title event sits at seq 2).
    assert [m["seq"] for m in provider.requests[1].messages] == [0, 4]


async def test_run_pending_revision_mismatch_returns() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider(automatic="all-prompts")
    service.title_service.register(provider)
    _user(session, service, "Explain quantum computing simply")
    _header(session, service, provider="prov", model="mod")
    # A newer message supersedes the started revision before its task runs.
    _user(session, service, "And now explain relativity")
    _header(session, service, provider="prov", model="mod")
    await service.title_service.drain()
    assert len(provider.requests) == 1
    assert [m["seq"] for m in provider.requests[0].messages] == [0, 2]


async def test_first_prompt_skips_seeded_sessions() -> None:
    _, service = _service()
    parent = service.create("parent")
    provider = _FakeProvider()
    service.title_service.register(provider)
    child = service.fork(parent, child_session_id="child")
    assert child.header["isSeeded"] is True
    _user(child, service, "Explain quantum computing simply")
    await _pump()
    _header(child, service, provider="prov", model="mod")
    await service.title_service.drain()
    assert provider.requests == []
    assert service.title_service.get(child)["source"] == {"kind": "fallback"}


async def test_first_prompt_skips_subagent_sessions() -> None:
    _, service = _service()
    provider = _FakeProvider()
    service.title_service.register(provider)
    session = service.store.create(
        "sub1",
        options={"meta": {"origin": "subagent", "createdAt": 1700000000000}},
    )
    assert session.header["origin"] == "subagent"
    _user(session, service, "Explain quantum computing simply")
    await _pump()
    _header(session, service, provider="prov", model="mod")
    await service.title_service.drain()
    assert provider.requests == []
    assert service.title_service.get(session)["source"] == {"kind": "fallback"}


async def test_ineligible_message_schedules_nothing() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider()
    service.title_service.register(provider)
    _user(session, service, "   ", source_kind="user")
    _user(session, service, "Injected context", source_kind="system")
    await _pump()
    assert service.title_service.get(session) is None
    assert provider.requests == []
    assert _title_events(session) == []


async def test_non_live_session_gets_no_title() -> None:
    _, service = _service()
    provider = _FakeProvider()
    service.title_service.register(provider)
    detached = Session.create("detached")
    service.log_event(
        detached,
        EventType.USER_MESSAGE,
        {
            "content": [{"type": "text", "text": "Explain quantum computing"}],
            "source": {"kind": "user"},
        },
    )
    await _pump()
    assert service.title_service.get(detached) is None
    assert provider.requests == []


async def test_header_before_watermark_does_not_start_pending() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider()
    service.title_service.register(provider)
    _header(session, service, provider="prov", model="mod")
    _user(session, service, "Explain quantum computing simply")
    await _pump()
    assert provider.requests == []
    _header(session, service, provider="prov", model="mod")
    await service.title_service.drain()
    assert len(provider.requests) == 1


async def test_header_without_route_keeps_pending() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider()
    service.title_service.register(provider)
    _user(session, service, "Explain quantum computing simply")
    await _pump()
    service.log_event(
        session,
        EventType.REQUEST_HEADER,
        {"header": {"config": {}}, "reason": "initial"},
    )
    await _pump()
    assert provider.requests == []
    _header(session, service, provider="prov", model="mod")
    await service.title_service.drain()
    assert len(provider.requests) == 1


async def test_provider_failure_leaves_fallback_and_warns(caplog: Any) -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider(fail=RuntimeError("model down"))
    service.title_service.register(provider)
    _user(session, service, "How do I reset my password for the dashboard?")
    _header(session, service, provider="prov", model="mod")
    with caplog.at_level("WARNING", logger="harness.session.title"):
        await service.title_service.drain()
    assert len(provider.requests) == 1
    snapshot = service.title_service.get(session)
    assert snapshot["title"] == "How do I reset my"
    assert snapshot["source"] == {"kind": "fallback"}
    assert "automatic title generation failed" in caplog.text


# ---------------------------------------------------------------------------
# Provider result validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "result",
    [
        None,
        "nope",
        {"messageSeqs": [0]},
        {"title": 42, "messageSeqs": [0]},
    ],
)
def test_validate_title_result_rejects_wrong_shape(result: Any) -> None:
    messages = [
        {"seq": 0, "text": "first"},
        {"seq": 1, "text": "second"},
    ]
    with pytest.raises(TypeError):
        validate_title_result(result, messages, 80)


@pytest.mark.parametrize(
    "result",
    [
        {"title": "   ", "messageSeqs": [0]},
        {"title": "Good", "messageSeqs": []},
        {"title": "Good", "messageSeqs": [99]},
        {"title": "Good", "messageSeqs": [1, 0]},
        {"title": "Good", "messageSeqs": [0, 0]},
        {"title": "Good", "messageSeqs": [-1]},
        {"title": "Good", "messageSeqs": [True]},
        {"title": "Good", "messageSeqs": [0], "model": {"provider": "p"}},
        {"title": "Good", "messageSeqs": [0], "model": {"model": "m"}},
        {"title": "Good", "messageSeqs": [0], "model": "nope"},
    ],
)
def test_validate_title_result_rejects(result: Any) -> None:
    messages = [
        {"seq": 0, "text": "first"},
        {"seq": 1, "text": "second"},
    ]
    with pytest.raises(ValueError):
        validate_title_result(result, messages, 80)


def test_validate_title_result_accepts_subset_in_order() -> None:
    messages = [
        {"seq": 0, "text": "first"},
        {"seq": 1, "text": "second"},
        {"seq": 5, "text": "third"},
    ]
    accepted = validate_title_result(
        {
            "title": "  Good   Title  ",
            "messageSeqs": [1, 5],
            "model": {"provider": "p", "model": "m"},
        },
        messages,
        80,
    )
    assert accepted == {
        "title": "Good Title",
        "messageSeqs": [1, 5],
        "model": {"provider": "p", "model": "m"},
    }


def test_validate_title_result_truncates_to_max_bytes() -> None:
    messages = [{"seq": 0, "text": "first"}]
    accepted = validate_title_result(
        {"title": "Hello world", "messageSeqs": [0]}, messages, 5
    )
    assert accepted["title"] == "Hello"


@pytest.mark.parametrize(
    "result",
    [
        {"title": "   ", "messageSeqs": [0]},
        {"title": "Good", "messageSeqs": [99]},
        {"title": "Good", "messageSeqs": []},
    ],
)
async def test_invalid_provider_result_never_commits(result: dict[str, Any]) -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider(result=result)
    service.title_service.register(provider)
    _user(session, service, "How do I reset my password for the dashboard?")
    await _pump()
    with pytest.raises(ValueError):
        await service.title_service.refresh(session)
    events = _title_events(session)
    assert len(events) == 1
    assert events[0].payload["source"] == {"kind": "fallback"}


async def test_provider_result_with_model_is_recorded() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider(
        result={
            "title": "Model Title",
            "messageSeqs": [0],
            "model": {"provider": "prov", "model": "mod"},
        }
    )
    service.title_service.register(provider)
    _user(session, service, "Explain quantum computing simply")
    snapshot = await service.title_service.refresh(session)
    assert snapshot is not None
    assert snapshot["source"] == {
        "kind": "provider",
        "provider": "fake-provider",
        "model": {"provider": "prov", "model": "mod"},
    }


# ---------------------------------------------------------------------------
# Pinning regression: a renamed session keeps its title through later turns.
# ---------------------------------------------------------------------------


async def test_rename_pins_against_automatic_titling() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider()
    provider.gate = asyncio.Event()
    service.title_service.register(provider)
    _user(session, service, "Explain quantum computing simply")
    _header(session, service, provider="prov", model="mod")
    await _pump()
    assert len(provider.requests) == 1
    # Rename while the provider call is still in flight.
    service.title_service.rename(session, "My Pinned Title")
    provider.gate.set()
    await service.title_service.drain()
    assert service.title_service.get(session)["title"] == "My Pinned Title"
    assert service.title_service.get(session)["source"] == {"kind": "user"}
    # Subsequent turns schedule nothing and change nothing.
    _user(session, service, "And now explain relativity")
    _header(session, service, provider="prov", model="mod")
    await service.title_service.drain()
    _user(session, service, "And now explain black holes")
    _header(session, service, provider="prov", model="mod")
    await service.title_service.drain()
    assert len(provider.requests) == 1
    assert service.title_service.get(session)["title"] == "My Pinned Title"
    assert len(_title_events(session)) == 2  # fallback + rename


# ---------------------------------------------------------------------------
# Disposal and late results.
# ---------------------------------------------------------------------------


async def test_provider_disposal_aborts_active_work() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider()
    provider.gate = asyncio.Event()
    dispose = service.title_service.register(provider)
    _user(session, service, "Explain quantum computing simply")
    _header(session, service, provider="prov", model="mod")
    await _pump()
    assert len(provider.requests) == 1
    dispose()
    provider.gate.set()
    await service.title_service.drain()
    snapshot = service.title_service.get(session)
    assert snapshot["source"] == {"kind": "fallback"}
    assert len(_title_events(session)) == 1


async def test_provider_disposal_aborts_pending_work() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider()
    dispose = service.title_service.register(provider)
    _user(session, service, "Explain quantum computing simply")
    await _pump()
    assert len(provider.requests) == 0
    dispose()
    _header(session, service, provider="prov", model="mod")
    await service.title_service.drain()
    assert provider.requests == []


async def test_dispose_service_aborts_in_flight_work() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider()
    provider.gate = asyncio.Event()
    service.title_service.register(provider)
    _user(session, service, "Explain quantum computing simply")
    _header(session, service, provider="prov", model="mod")
    await _pump()
    assert len(provider.requests) == 1
    service.title_service.dispose()
    service.title_service.dispose()
    provider.gate.set()
    await service.title_service.drain()
    assert service.title_service.get(session)["source"] == {"kind": "fallback"}


async def test_session_close_aborts_in_flight_work() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider()
    provider.gate = asyncio.Event()
    service.title_service.register(provider)
    _user(session, service, "Explain quantum computing simply")
    _header(session, service, provider="prov", model="mod")
    await _pump()
    assert len(provider.requests) == 1
    assert service.close("s1") is True
    provider.gate.set()
    await service.title_service.drain()
    assert service.title_service.get(session)["source"] == {"kind": "fallback"}


async def test_close_before_fallback_warns_and_does_not_crash(caplog: Any) -> None:
    _, service = _service()
    session = service.create("s1")
    _user(session, service, "How do I reset my password?")
    assert service.close("s1") is True
    with caplog.at_level("WARNING", logger="harness.session.title"):
        await service.title_service.drain()
    assert "deferred title work failed" in caplog.text


async def test_dispose_before_fallback_is_silent(caplog: Any) -> None:
    _, service = _service()
    session = service.create("s1")
    _user(session, service, "How do I reset my password?")
    service.title_service.dispose()
    with caplog.at_level("WARNING", logger="harness.session.title"):
        await service.title_service.drain()
    assert "deferred title work failed" not in caplog.text
    assert _title_events(session) == []


async def test_dispose_removes_listeners() -> None:
    _, service = _service()
    session = service.create("s1")
    service.title_service.dispose()
    _user(session, service, "How do I reset my password?")
    await _pump()
    assert service.title_service.get(session) is None


async def test_dispose_with_pending_work_aborts_nothing_live() -> None:
    _, service = _service()
    session = service.create("s1")
    provider = _FakeProvider()
    service.title_service.register(provider)
    _user(session, service, "Explain quantum computing simply")
    # Pending work exists but no generation started; disposal is quiet.
    service.title_service.dispose()
    await service.title_service.drain()
    assert provider.requests == []
    with pytest.raises(RuntimeError, match="disposed"):
        service.title_service.rename(session, "Late Rename")


async def test_session_event_listener_ignores_after_dispose() -> None:
    # A listener registered before the title service runs first in the
    # emit snapshot; disposing there must leave the title listener a
    # no-op instead of scheduling work.
    ctx = Context()
    holder: dict[str, Any] = {}

    def dispose_first(session_arg: Session, event: SessionEvent) -> None:
        holder["service"].title_service.dispose()

    ctx.on("session/event", dispose_first)
    service = SessionService(ctx, {})
    holder["service"] = service
    session = service.create("s1")
    _user(session, service, "How do I reset my password?")
    await service.title_service.drain()
    assert service.title_service.get(session) is None
    assert _title_events(session) == []


async def test_drain_with_no_work_returns() -> None:
    _, service = _service()
    await service.title_service.drain()


# ---------------------------------------------------------------------------
# Sync (no running loop) deferred fallback via the timer path.
# ---------------------------------------------------------------------------


def test_sync_append_lands_fallback_via_timer() -> None:
    _, service = _service()
    session = service.create("s1")
    _user(session, service, "Hello there world")
    deadline = time.monotonic() + 5
    snapshot = None
    while time.monotonic() < deadline:
        snapshot = service.title_service.get(session)
        if snapshot is not None:
            break
        time.sleep(0.01)
    assert snapshot is not None
    assert snapshot["title"] == "Hello there world"
    assert snapshot["source"] == {"kind": "fallback"}


def test_sync_close_before_fallback_warns(caplog: Any) -> None:
    _, service = _service()
    session = service.create("s1")
    _user(session, service, "How do I reset my password?")
    assert service.close("s1") is True
    deadline = time.monotonic() + 5
    with caplog.at_level("WARNING", logger="harness.session.title"):
        while (
            time.monotonic() < deadline
            and "deferred title work failed" not in caplog.text
        ):
            time.sleep(0.01)
    assert "deferred title work failed" in caplog.text


def test_sync_drain_joins_timer_work() -> None:
    _, service = _service()
    session = service.create("s1")
    _user(session, service, "Hello there world")
    asyncio.run(service.title_service.drain())
    snapshot = service.title_service.get(session)
    assert snapshot is not None
    assert snapshot["title"] == "Hello there world"


def test_sync_dispose_silences_timer_work() -> None:
    _, service = _service()
    session = service.create("s1")
    _user(session, service, "How do I reset my password?")
    service.title_service.dispose()
    # The timer fires after disposal; its work must never run.
    time.sleep(0.2)
    assert service.title_service.get(session) is None
    assert _title_events(session) == []


# ---------------------------------------------------------------------------
# Title survives a restart (log-backed, not in memory).
# ---------------------------------------------------------------------------


async def test_title_survives_restart(tmp_path: Path) -> None:
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    session = service.create("s1")
    _user(session, service, "How do I reset my password for the dashboard?")
    await service.title_service.drain()
    assert service.title_service.get(session)["title"] == "How do I reset my"
    assert service.close("s1") is True

    ctx2 = Context()
    service2 = SessionService(ctx2, {"base_storage_dir": tmp_path})
    loaded = service2.load("s1")
    assert loaded is not None
    snapshot = service2.title_service.get(loaded)
    assert snapshot is not None
    assert snapshot["title"] == "How do I reset my"
    assert snapshot["source"] == {"kind": "fallback"}
    assert snapshot["messageSeqs"] == [0]
    # The restored session still pins and refreshes from the log.
    service2.title_service.rename(loaded, "Restored Name")
    assert service2.title_service.get(loaded)["title"] == "Restored Name"


async def test_fallback_is_deterministic_over_first_message() -> None:
    _, service = _service()
    first = service.create("first")
    second = service.create("second")
    _user(first, service, "How do I reset my password for the dashboard?")
    _user(second, service, "How do I reset my password for the dashboard?")
    await service.title_service.drain()
    # Only the derived fields are deterministic; eventSeq/updatedAt differ.
    for field in ("title", "messageSeqs", "source"):
        assert (
            service.title_service.get(first)[field]
            == (service.title_service.get(second)[field])
        )


async def test_append_fallback_underivable_returns_none() -> None:
    # A one-byte fallback budget cannot hold the two-byte first word,
    # so no title event lands at all.
    _, service = _service(
        {
            "title_fallback_max_words": 5,
            "title_fallback_max_bytes": 1,
            "title_max_title_bytes": 1,
        }
    )
    session = service.create("s1")
    _user(session, service, "é")
    await service.title_service.drain()
    assert service.title_service.get(session) is None
    assert _title_events(session) == []


# ---------------------------------------------------------------------------
# llm/stream trigger: a matching agent-loop request starts pending work.
# ---------------------------------------------------------------------------


async def _stream_harness(tmp_path: Path) -> Any:
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    ctx.llm.set_active_provider("mock")
    return ctx


async def test_llm_stream_starts_pending(tmp_path: Path) -> None:
    ctx = await _stream_harness(tmp_path)
    session = ctx.sessions.create("s1")
    title_service = ctx.sessions.title_service
    provider = _FakeProvider()
    title_service.register(provider)
    _header(session, ctx.sessions, provider="mock", model="model-x")
    _user(session, ctx.sessions, "Explain quantum computing simply")
    await _pump()
    assert provider.requests == []
    chunks = []
    async for chunk in ctx.llm.stream(
        messages=[], model="mock/model-x", session_id=session.session_id
    ):
        chunks.append(chunk)
    assert chunks
    await title_service.drain()
    assert len(provider.requests) == 1
    assert provider.requests[0].route == {"provider": "mock", "model": "model-x"}
    assert title_service.get(session)["title"] == "Provider Title"


async def test_llm_stream_route_mismatch_does_not_start(tmp_path: Path) -> None:
    ctx = await _stream_harness(tmp_path)
    session = ctx.sessions.create("s1")
    title_service = ctx.sessions.title_service
    provider = _FakeProvider()
    title_service.register(provider)
    _header(session, ctx.sessions, provider="mock", model="model-x")
    _user(session, ctx.sessions, "Explain quantum computing simply")
    await _pump()
    async for _ in ctx.llm.stream(
        messages=[], model="mock/other-model", session_id=session.session_id
    ):
        pass
    await title_service.drain()
    assert provider.requests == []


async def test_llm_stream_without_model_does_not_start(tmp_path: Path) -> None:
    ctx = await _stream_harness(tmp_path)
    session = ctx.sessions.create("s1")
    title_service = ctx.sessions.title_service
    provider = _FakeProvider()
    title_service.register(provider)
    _header(session, ctx.sessions, provider="mock", model="model-x")
    _user(session, ctx.sessions, "Explain quantum computing simply")
    await _pump()
    async for _ in ctx.llm.stream(messages=[], session_id=session.session_id):
        pass
    await title_service.drain()
    assert provider.requests == []


async def test_llm_stream_unknown_session_passes_through(tmp_path: Path) -> None:
    ctx = await _stream_harness(tmp_path)
    session = ctx.sessions.create("s1")
    title_service = ctx.sessions.title_service
    provider = _FakeProvider()
    title_service.register(provider)
    _header(session, ctx.sessions, provider="mock", model="model-x")
    _user(session, ctx.sessions, "Explain quantum computing simply")
    await _pump()
    chunks = []
    async for chunk in ctx.llm.stream(
        messages=[], model="mock/model-x", session_id="unknown-session"
    ):
        chunks.append(chunk)
    assert chunks
    await title_service.drain()
    assert provider.requests == []


async def test_llm_stream_listener_ignores_after_dispose() -> None:
    # A waterfall listener registered before the title service runs
    # first in the dispatch snapshot; disposing there must leave the
    # title listener a pass-through instead of starting work.
    ctx = Context()
    holder: dict[str, Any] = {}

    async def dispose_first(options: dict[str, Any], next_fn: Any) -> Any:
        holder["service"].title_service.dispose()
        return await next_fn(options)

    ctx.on("llm/stream", dispose_first, mode=EventMode.WATERFALL)
    service = SessionService(ctx, {})
    holder["service"] = service
    llm = LLMService(ctx)
    llm.register_provider("mock", MockLLMProvider())
    session = service.create("s1")
    chunks = []
    async for chunk in llm.stream(
        messages=[], model="mock/model-x", session_id=session.session_id
    ):
        chunks.append(chunk)
    assert chunks
    assert service.title_service.get(session) is None
