"""Log-backed session titles: deterministic fallback, pinning, provider contract.

The session log is the title source of truth. Every accepted title is a
``session/title`` event carrying ``{title, messageSeqs, source}`` and the
current title is the latest-wins fold over the log, so a title survives a
restart. The event is log-only: it never enters the model surface.

A title has one of three sources: ``fallback`` (the deterministic title
derived from the first eligible human message), ``provider`` (a registered
title provider, optionally with the model route that produced it), or
``user`` (an explicit rename). A user title pins: while it stands, no
automatic revision is scheduled, and only an explicit ``refresh`` unpins
it by re-deriving over it.

The service takes the ``title`` and ``titleInput`` projection handles
from the session service. It never registers those units itself: a second
registration of the same key at a different state version throws.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
import threading
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, Protocol, TypedDict

from anima import Context, Disposer, EventMode

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.llm.slug import parse_model_slug
from magitech_harness.session.events import SessionEvent
from magitech_harness.session.projection_registry import ProjectionHandle
from magitech_harness.session.store import Session, SessionStore

logger = logging.getLogger("harness.session.title")


class SessionTitleInvalidError(ValueError):
    """Rejection of an explicit user title whose text normalizes to empty.

    The one ``rename`` failure that blames the input; liveness and
    disposal failures stay plain errors.
    """

    name = "SessionTitleInvalidError"


# ---------------------------------------------------------------------------
# Title-text normalization (escape sequences, controls, UTF-8 byte budget).
# ---------------------------------------------------------------------------

_OSC_SEQUENCE = re.compile(
    r"(?:\x1b\]|\x9d)(?:(?!\x07|\x1b\\)[\s\S])*(?:\x07|\x1b\\|\Z)"
)
_CSI_SEQUENCE = re.compile(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]")
_ESC_SEQUENCE = re.compile(r"\x1b[@-_]")
_CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_DIRECTIONAL_CONTROL = re.compile(
    r"[\u200b\u200e\u200f\u202a-\u202e\u2060-\u2064\u2066-\u206f\ufeff]"
)
_WHITESPACE_RUN = re.compile(r"\s+")


def clean_title_text(text: str) -> str:
    """Strip escape sequences and controls; collapse whitespace runs; trim."""
    cleaned = _OSC_SEQUENCE.sub("", text)
    cleaned = _CSI_SEQUENCE.sub("", cleaned)
    cleaned = _ESC_SEQUENCE.sub("", cleaned)
    cleaned = _CONTROL_CHARACTER.sub("", cleaned)
    cleaned = _DIRECTIONAL_CONTROL.sub("", cleaned)
    return _WHITESPACE_RUN.sub(" ", cleaned).strip()


def truncate_title_utf8(text: str, max_bytes: int) -> str:
    """Longest leading code-point prefix of ``text`` within ``max_bytes`` bytes.

    Never splits a code point: a character whose bytes would exceed the
    budget ends the prefix.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("session-title: max_bytes must be a positive integer")
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    # A cut inside a multi-byte character leaves an incomplete trailing
    # sequence, which decoding ignores; no code point is ever split.
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def normalize_session_title(text: str, max_bytes: int) -> str:
    """Clean one title candidate and enforce its UTF-8 byte budget."""
    return truncate_title_utf8(clean_title_text(text), max_bytes).rstrip()


def fallback_session_title(text: str, max_words: int, max_bytes: int) -> str:
    """Deterministic title from the leading ``max_words`` words of ``text``.

    The words are taken after cleaning, then the byte cap applies.
    """
    if isinstance(max_words, bool) or not isinstance(max_words, int) or max_words <= 0:
        raise ValueError("session-title: max_words must be a positive integer")
    words = [word for word in clean_title_text(text).split(" ") if word][:max_words]
    return truncate_title_utf8(" ".join(words), max_bytes).rstrip()


def eligible_title_message(event: SessionEvent) -> dict[str, Any] | None:
    """The eligible human message of one event, or ``None``.

    Eligible iff the event is a ``user/message`` whose source kind is
    ``"user"`` and whose joined text blocks clean to non-empty. The
    value carries the raw joined block text; cleaning is only the
    emptiness test.
    """
    if event.type != "user/message":
        return None
    source = event.payload.get("source")
    if not isinstance(source, dict) or source.get("kind") != "user":
        return None
    content = event.payload.get("content")
    if not isinstance(content, list):
        return None
    text = "\n".join(
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )
    if clean_title_text(text) == "":
        return None
    return {"seq": event.seq, "text": text}


# ---------------------------------------------------------------------------
# Configuration: required, no library defaults.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TitleConfig:
    """Required title limits. The composition layer supplies the values."""

    fallback_max_words: int
    fallback_max_bytes: int
    max_title_bytes: int


def _positive_title_limit(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"session-title: {name} must be a positive integer, got {value!r}"
        )
    return value


def resolve_title_config(config: dict[str, Any] | None) -> TitleConfig:
    """Validate raw title config into immutable limits.

    Raises ``ValueError`` when the config is missing, carries a
    non-positive-integer limit, or sets ``fallback_max_bytes`` above
    ``max_title_bytes``.
    """
    if config is None or not isinstance(config, dict):
        raise ValueError("session-title: configuration is required")
    fallback_max_words = _positive_title_limit(config, "fallback_max_words")
    fallback_max_bytes = _positive_title_limit(config, "fallback_max_bytes")
    max_title_bytes = _positive_title_limit(config, "max_title_bytes")
    if fallback_max_bytes > max_title_bytes:
        raise ValueError(
            "session-title: fallback_max_bytes must not exceed max_title_bytes"
        )
    return TitleConfig(
        fallback_max_words=fallback_max_words,
        fallback_max_bytes=fallback_max_bytes,
        max_title_bytes=max_title_bytes,
    )


# ---------------------------------------------------------------------------
# Fold helpers: the log is the title source of truth.
# ---------------------------------------------------------------------------


def title_snapshot_from_event(event: SessionEvent) -> dict[str, Any] | None:
    """The snapshot for one ``session/title`` event, or ``None``.

    Returns ``None`` for an event of another type, or one carrying no
    string title. A non-dict source or a non-list ``messageSeqs`` degrades
    to an empty source / empty seqs rather than failing the fold.
    """
    if event.type != "session/title":
        return None
    title = event.payload.get("title")
    if not isinstance(title, str):
        return None
    message_seqs = event.payload.get("messageSeqs")
    source = event.payload.get("source")
    return {
        "title": title,
        "messageSeqs": list(message_seqs) if isinstance(message_seqs, list) else [],
        "source": copy.deepcopy(source) if isinstance(source, dict) else {},
        "eventSeq": event.seq,
        "updatedAt": int(event.timestamp * 1000),
    }


def fold_session_title(events: list[SessionEvent]) -> dict[str, Any] | None:
    """Latest folded title snapshot over a log, or ``None`` before any title.

    Latest-wins by log order: the last ``session/title`` event carrying a
    string title wins.
    """
    for event in reversed(events):
        snapshot = title_snapshot_from_event(event)
        if snapshot is not None:
            return snapshot
    return None


def collect_title_messages(
    events: list[SessionEvent], through_seq: int | None = None
) -> list[dict[str, Any]]:
    """Eligible human messages in seq order, bounded at ``through_seq``.

    The full eligible prefix is only materialized for one provider
    generation, so it is scanned from the log at execution time rather
    than retained by the ``titleInput`` projection.
    """
    messages: list[dict[str, Any]] = []
    for event in events:
        if through_seq is not None and event.seq > through_seq:
            break
        message = eligible_title_message(event)
        if message is not None:
            messages.append(message)
    return messages


def _route_from_header(header: Any) -> dict[str, str] | None:
    """The ``{provider, model}`` route of one request-header record."""
    config = header.get("config") if isinstance(header, dict) else None
    if not isinstance(config, dict):
        return None
    provider = config.get("provider")
    model = config.get("model")
    if isinstance(provider, str) and provider and isinstance(model, str) and model:
        return {"provider": provider, "model": model}
    return None


def current_route(session: Session) -> dict[str, str] | None:
    """Exact current logged main-request route, or ``None`` when none is recorded.

    Folds the latest ``request/header`` event; a header without a
    provider/model pair yields no route.
    """
    for event in reversed(session.events):
        if event.type != "request/header":
            continue
        return _route_from_header(event.payload.get("header"))
    return None


# ---------------------------------------------------------------------------
# Provider contract.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TitleProviderRequest:
    """Immutable input supplied to one title-provider call."""

    session: Session
    messages: tuple[dict[str, Any], ...]
    route: dict[str, str] | None = None
    current_title: dict[str, Any] | None = None
    signal: AbortSignal = field(default_factory=AbortSignal)


class TitleProviderResult(TypedDict):
    """Provider output before service-owned normalization and log acceptance."""

    title: str
    messageSeqs: list[int]
    model: NotRequired[dict[str, str]]


class TitleProvider(Protocol):
    """One optional asynchronous title implementation registered with the service."""

    id: str
    automatic: Literal["first-prompt", "all-prompts"]

    async def generate(self, request: TitleProviderRequest) -> TitleProviderResult:
        """Produce one title revision from the request's message snapshot."""
        ...


def validate_title_result(
    result: Any, messages: list[dict[str, Any]], max_title_bytes: int
) -> dict[str, Any]:
    """Validate and normalize one provider result against its request messages.

    The title is normalized against ``max_title_bytes`` and must be
    non-empty; ``messageSeqs`` must be non-empty, unique, ordered, and
    drawn from the request's messages; an optional ``model`` must carry
    non-empty ``provider`` and ``model`` strings. Raises ``TypeError``
    for a wrongly shaped result and ``ValueError`` for an invalid value;
    a rejected result is never committed.
    """
    if not isinstance(result, dict):
        raise TypeError("session-title provider returned an invalid result")
    raw_title = result.get("title")
    if not isinstance(raw_title, str):
        raise TypeError("session-title provider title must be a string")
    title = normalize_session_title(raw_title, max_title_bytes)
    if title == "":
        raise ValueError("session-title provider returned an empty title")
    raw_seqs = result.get("messageSeqs")
    if not isinstance(raw_seqs, list) or not raw_seqs:
        raise ValueError(
            "session-title provider must identify at least one source message seq"
        )
    order = {message["seq"]: index for index, message in enumerate(messages)}
    message_seqs: list[int] = []
    previous = -1
    for seq in raw_seqs:
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise ValueError(
                "session-title provider messageSeqs must be unique, ordered "
                "seqs from the request"
            )
        index = order.get(seq)
        if index is None or index <= previous:
            raise ValueError(
                "session-title provider messageSeqs must be unique, ordered "
                "seqs from the request"
            )
        message_seqs.append(seq)
        previous = index
    raw_model = result.get("model")
    model: dict[str, str] | None = None
    if raw_model is not None:
        if (
            not isinstance(raw_model, dict)
            or not isinstance(raw_model.get("provider"), str)
            or not raw_model["provider"]
            or not isinstance(raw_model.get("model"), str)
            or not raw_model["model"]
        ):
            raise ValueError(
                "session-title provider result model must contain non-empty "
                "provider and model strings"
            )
        model = {"provider": raw_model["provider"], "model": raw_model["model"]}
    accepted: dict[str, Any] = {"title": title, "messageSeqs": message_seqs}
    if model is not None:
        accepted["model"] = model
    return accepted


# ---------------------------------------------------------------------------
# Service.
# ---------------------------------------------------------------------------


@dataclass
class _ProviderRegistration:
    """One exact provider registration generation."""

    provider: TitleProvider
    closing: bool = False


@dataclass
class _PendingWork:
    """Automatic work waiting for its exact main-request route."""

    registration: _ProviderRegistration
    revision: int
    through_seq: int


@dataclass
class _ActiveWork:
    """Provider call currently allowed to commit for one session."""

    registration: _ProviderRegistration
    revision: int
    through_seq: int
    signal: AbortSignal
    upstream_disposer: Callable[[], None] | None = None


@dataclass
class _SessionWork:
    """Mutable concurrency state scoped to one live session."""

    revision: int = 0
    pending: _PendingWork | None = None
    active: _ActiveWork | None = None


class SessionTitleService:
    """Log-backed session title service.

    Owns the ``title`` and ``titleInput`` projection handles handed over
    by the session service, the sole optional title provider, and the
    automatic scheduling that derives titles from eligible human
    messages. A session append cannot reenter while another append is
    being published, so fallback work scheduled from a ``session/event``
    listener runs deferred: on the running event loop when one exists,
    otherwise on a short-delay timer thread.
    """

    # Delay before deferred title work runs without a running loop.
    # The outer append has returned long before this fires, so the
    # deferred append no longer reenters the publish.
    _DEFER_DELAY_S = 0.05

    def __init__(
        self,
        ctx: Context,
        store: SessionStore,
        title_handle: ProjectionHandle[str | None],
        title_input_handle: ProjectionHandle[dict[str, Any]],
        config: dict[str, Any] | None,
    ) -> None:
        """Build the service; ``config`` is required and has no defaults here.

        The composition layer supplies the shipped values. Raises
        ``ValueError`` when the config is missing or invalid.
        """
        self._ctx = ctx
        self._store = store
        self._title_handle = title_handle
        self._title_input_handle = title_input_handle
        self._config = resolve_title_config(config)
        self._registration: _ProviderRegistration | None = None
        self._work: dict[Session, _SessionWork] = {}
        self._disposed = False
        self._in_flight: set[asyncio.Task[Any]] = set()
        self._timers: set[threading.Timer] = set()
        self._disposers: list[Disposer] = [
            ctx.on("session/event", self._on_session_event),
            ctx.on("session/disposed", self._on_session_disposed),
            ctx.on("llm/stream", self._on_llm_stream, mode=EventMode.WATERFALL),
        ]

    def dispose(self) -> None:
        """Remove all listeners and abort pending work. Idempotent.

        Late results never commit: in-flight work re-checks currency
        before appending, and disposal aborts its signal.
        """
        if self._disposed:
            return
        self._disposed = True
        for disposer in self._disposers:
            disposer()
        self._disposers.clear()
        if self._registration is not None:
            self._registration.closing = True
        self._registration = None
        for state in self._work.values():
            if state.active is not None:
                state.active.signal.abort("session-title service disposed")
            state.pending = None
        self._work.clear()
        for task in list(self._in_flight):
            task.cancel()
        # Timers are not cancelled: a timer that fires after disposal sees
        # the disposed flag in _run_deferred_in_thread and closes its
        # coroutine without running it.

    async def drain(self) -> None:
        """Await every in-flight title task until none remain.

        Done tasks are discarded explicitly instead of relying on their
        done-callbacks: awaiting an already-finished task never yields to
        the loop, so a loop that only gathers would spin forever on a
        finished task whose callback has not run yet. Timer-deferred work
        scheduled without a running loop is joined the same way, so a
        caller can wait for quiescence after sync appends too.
        """
        while True:
            self._in_flight = {task for task in self._in_flight if not task.done()}
            timers = list(self._timers)
            if not self._in_flight and not timers:
                return
            if self._in_flight:
                await asyncio.gather(*self._in_flight, return_exceptions=True)
            else:
                await asyncio.to_thread(self._join_timers, timers)

    @staticmethod
    def _join_timers(timers: list[threading.Timer]) -> None:
        for timer in timers:
            timer.join()

    def get(self, session: Session) -> dict[str, Any] | None:
        """Latest folded title snapshot, or ``None`` before any title lands."""
        return fold_session_title(session.events)

    def rename(self, session: Session, title: str) -> dict[str, Any]:
        """Accept an explicit user title, pinning it against automatic titling.

        Normalizes the input; empty after normalization raises
        :class:`SessionTitleInvalidError`. A non-live session raises a
        plain error. Appends a ``session/title`` event with the ``user``
        source, superseding in-flight automatic generation.
        """
        self._assert_active()
        if not self._is_live(session):
            raise ValueError(
                f'session "{session.session_id}" is not live in this store'
            )
        normalized = normalize_session_title(title, self._config.max_title_bytes)
        if normalized == "":
            raise SessionTitleInvalidError(
                "session title must contain visible characters"
            )
        state = self._state_for(session)
        self._supersede(state, "user rename superseded automatic title generation")
        event = session.append(
            "session/title",
            {"title": normalized, "messageSeqs": [], "source": {"kind": "user"}},
        )
        snapshot = title_snapshot_from_event(event)
        assert snapshot is not None
        return snapshot

    async def refresh(
        self, session: Session, signal: AbortSignal | None = None
    ) -> dict[str, Any] | None:
        """Explicitly retry the provider, or materialize the fallback.

        This is the unpin even without a provider: a standing user title
        does not short-circuit the fallback, so re-deriving appends the
        fallback over it when one is derivable. Returns the latest
        accepted title, or ``None`` when no eligible text exists.
        """
        if signal is not None:
            signal.throw_if_aborted()
        self._assert_active()
        if not self._is_live(session):
            raise ValueError(
                f'session "{session.session_id}" is not live in this store'
            )
        registration = self._registration
        title_input = self._title_input_of(session)
        if (
            registration is None
            or registration.closing
            or title_input["last_seq"] is None
        ):
            current = self.get(session)
            first = title_input["first"]
            if (
                current is not None
                and current.get("source", {}).get("kind") == "user"
                and first is not None
            ):
                self._append_fallback(session, first)
                if signal is not None:
                    signal.throw_if_aborted()
                return self.get(session)
            fallback = self._ensure_fallback(session)
            if signal is not None:
                signal.throw_if_aborted()
            return fallback
        state = self._state_for(session)
        revision = self._supersede(
            state, "explicit title refresh superseded older generation"
        )
        work = self._activate(
            _PendingWork(registration, revision, title_input["last_seq"]),
            state,
            signal,
        )
        route = current_route(session)
        return await self._run_provider(session, work, route)

    def register(self, provider: TitleProvider) -> Disposer:
        """Register the sole optional title provider.

        Disposal aborts its pending and active work; a replacement may
        register once disposal has started, and the closing provider's
        late results never commit. A second live registration throws.
        Returns a disposer for the registration.
        """
        self._assert_active()
        self._validate_provider(provider)
        if self._registration is not None and not self._registration.closing:
            raise ValueError(
                "session-title provider "
                f'"{self._registration.provider.id}" is already registered'
            )
        registration = _ProviderRegistration(provider=provider)
        self._registration = registration

        def dispose() -> None:
            registration.closing = True
            for state in list(self._work.values()):
                if (
                    state.pending is not None
                    and state.pending.registration is registration
                ):
                    state.pending = None
                if (
                    state.active is not None
                    and state.active.registration is registration
                ):
                    state.active.signal.abort(
                        f'session-title provider "{provider.id}" was disposed'
                    )
            if self._registration is registration:
                self._registration = None

        return dispose

    # -- listeners ----------------------------------------------------------

    def _on_session_event(self, session: Session, event: SessionEvent) -> None:
        if self._disposed:
            return
        if event.type == "user/message":
            self._on_user_message(session, event)
        elif event.type == "request/header":
            self._on_request_header(session, event)

    def _on_session_disposed(self, session: Session) -> None:
        state = self._work.pop(session, None)
        if state is not None and state.active is not None:
            state.active.signal.abort("session disposed during title generation")

    async def _on_llm_stream(
        self,
        options: dict[str, Any],
        next_fn: Callable[..., Awaitable[Any]],
    ) -> Any:
        """Start unchanged-route pending work from a matching loop request."""
        if self._disposed:
            return await next_fn()
        session_id = options.get("session_id")
        if session_id is None:
            return await next_fn()
        session = self._store.get(session_id)
        if session is None:
            return await next_fn()
        state = self._work.get(session)
        if state is None or state.pending is None:
            return await next_fn()
        pending = state.pending
        model = options.get("model")
        if not isinstance(model, str) or not model:
            return await next_fn()
        provider_name, model_name = parse_model_slug(model)
        route = current_route(session)
        if (
            route is None
            or route["provider"] != provider_name
            or route["model"] != model_name
        ):
            return await next_fn()
        self._start_pending(session, state, pending, route)
        return await next_fn()

    def _on_user_message(self, session: Session, event: SessionEvent) -> None:
        if eligible_title_message(event) is None:
            return
        if not self._is_live(session):
            return
        # A user rename pins the title: no automatic revision may override it.
        snapshot = self.get(session)
        if snapshot is not None and snapshot.get("source", {}).get("kind") == "user":
            return
        registration = self._registration
        if registration is not None and not registration.closing:
            provider = registration.provider
            count = self._title_input_of(session)["count"]
            header = session.header
            should_schedule = provider.automatic == "all-prompts" or (
                header.get("isSeeded") is not True
                and header.get("origin") != "subagent"
                and count == 1
                and snapshot is None
            )
            if should_schedule:
                state = self._state_for(session)
                revision = self._supersede(
                    state, "newer user message superseded title generation"
                )
                state.pending = _PendingWork(registration, revision, event.seq)
        self._defer(self._deferred_fallback(session))

    def _on_request_header(self, session: Session, event: SessionEvent) -> None:
        # Reached only from _on_session_event, whose disposed guard covers
        # this listener; no second guard here.
        state = self._work.get(session)
        pending = state.pending if state is not None else None
        if pending is None or pending.through_seq >= event.seq:
            return
        assert state is not None
        route = _route_from_header(event.payload.get("header"))
        if route is None:
            return
        self._start_pending(session, state, pending, route)

    # -- provider execution -------------------------------------------------

    def _start_pending(
        self,
        session: Session,
        state: _SessionWork,
        pending: _PendingWork,
        route: dict[str, str],
    ) -> None:
        """Consume one pending revision and schedule its provider call."""
        state.pending = None
        self._defer(self._run_pending(session, state, pending, route))

    async def _run_pending(
        self,
        session: Session,
        state: _SessionWork,
        pending: _PendingWork,
        route: dict[str, str],
    ) -> None:
        if (
            self._disposed
            or self._registration is not pending.registration
            or pending.registration.closing
            or self._work.get(session) is not state
            or state.revision != pending.revision
        ):
            return
        work = self._activate(pending, state)
        try:
            await self._run_provider(session, work, route)
        except Exception as error:  # noqa: BLE001 - automatic generation never fails the turn
            if work.signal.is_aborted or self._disposed:
                return
            logger.warning(
                'session "%s": automatic title generation failed: %s',
                session.session_id,
                error,
            )

    async def _run_provider(
        self,
        session: Session,
        work: _ActiveWork,
        route: dict[str, str] | None,
    ) -> dict[str, Any] | None:
        """Execute and accept one current provider revision."""
        try:
            self._assert_current(session, work)
            # A title exists even if the provider fails.
            self._ensure_fallback(session)
            self._assert_current(session, work)
            # One snapshot supplies both inputs: messages stop at the
            # scheduled watermark, while the current title is unbounded.
            messages = collect_title_messages(session.events, work.through_seq)
            current_title = self.get(session)
            result = await work.registration.provider.generate(
                TitleProviderRequest(
                    session=session,
                    messages=tuple(messages),
                    route=route,
                    current_title=current_title,
                    signal=work.signal,
                )
            )
            self._assert_current(session, work)
            accepted = validate_title_result(
                result, messages, self._config.max_title_bytes
            )
            source: dict[str, Any] = {
                "kind": "provider",
                "provider": work.registration.provider.id,
            }
            if "model" in accepted:
                source["model"] = accepted["model"]
            event = session.append(
                "session/title",
                {
                    "title": accepted["title"],
                    "messageSeqs": list(accepted["messageSeqs"]),
                    "source": source,
                },
            )
            snapshot = title_snapshot_from_event(event)
            assert snapshot is not None
            return snapshot
        finally:
            state = self._work.get(session)
            if state is not None and state.active is work:
                state.active = None
                if work.upstream_disposer is not None:
                    work.upstream_disposer()

    # -- helpers ------------------------------------------------------------

    def _assert_active(self) -> None:
        if self._disposed:
            raise RuntimeError("session-title service disposed")

    def _assert_current(self, session: Session, work: _ActiveWork) -> None:
        """Fail unless this revision is still the live one for the session.

        Currency is enforced by the abort protocol: every state change
        (supersede, provider disposal, session close, service disposal)
        aborts the work signal first, so a live signal proves the
        revision is current. Checked at every await boundary.
        """
        self._assert_active()
        work.signal.throw_if_aborted()

    def _is_live(self, session: Session) -> bool:
        return self._store.get(session.session_id) is session

    def _state_for(self, session: Session) -> _SessionWork:
        state = self._work.get(session)
        if state is None:
            state = _SessionWork()
            self._work[session] = state
        return state

    def _title_input_of(self, session: Session) -> dict[str, Any]:
        return self._title_input_handle.state_of(session)

    def _supersede(self, state: _SessionWork, reason: str) -> int:
        if state.active is not None:
            state.active.signal.abort(reason)
        state.pending = None
        state.revision += 1
        return state.revision

    def _activate(
        self,
        pending: _PendingWork,
        state: _SessionWork,
        upstream: AbortSignal | None = None,
    ) -> _ActiveWork:
        signal = AbortSignal()
        upstream_disposer: Callable[[], None] | None = None
        if upstream is not None:
            upstream_disposer = upstream.on_abort(
                lambda: signal.abort("title generation cancelled")
            )
        work = _ActiveWork(
            registration=pending.registration,
            revision=pending.revision,
            through_seq=pending.through_seq,
            signal=signal,
            upstream_disposer=upstream_disposer,
        )
        state.active = work
        return work

    @staticmethod
    def _validate_provider(provider: Any) -> None:
        provider_id = getattr(provider, "id", None)
        if not isinstance(provider_id, str) or not provider_id:
            raise ValueError("session-title provider id must be a non-empty string")
        automatic = getattr(provider, "automatic", None)
        if automatic not in ("first-prompt", "all-prompts"):
            raise ValueError("session-title provider automatic mode is invalid")
        if not callable(getattr(provider, "generate", None)):
            raise TypeError(
                f'session-title provider "{provider_id}" requires generate()'
            )

    def _append_fallback(
        self, session: Session, first: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Derive and append the deterministic fallback over whatever stands.

        Synchronous on purpose: no await may separate derivation from
        append. An underivable fallback (empty after the caps) appends
        nothing.
        """
        title = fallback_session_title(
            first["text"],
            self._config.fallback_max_words,
            self._config.fallback_max_bytes,
        )
        if title == "":
            return None
        event = session.append(
            "session/title",
            {
                "title": title,
                "messageSeqs": [first["seq"]],
                "source": {"kind": "fallback"},
            },
        )
        snapshot = title_snapshot_from_event(event)
        assert snapshot is not None
        return snapshot

    def _ensure_fallback(self, session: Session) -> dict[str, Any] | None:
        """Create the first deterministic fallback if the session lacks a title."""
        self._assert_active()
        if not self._is_live(session):
            raise ValueError(
                f'session "{session.session_id}" is not live in this store'
            )
        current = self.get(session)
        if current is not None:
            return current
        first = self._title_input_of(session)["first"]
        if first is None:
            return None
        return self._append_fallback(session, first)

    async def _deferred_fallback(self, session: Session) -> None:
        self._ensure_fallback(session)

    def _defer(self, coro: Coroutine[Any, Any, None]) -> None:
        """Schedule deferred title work after the current publish completes.

        On a running event loop the work is a task; without one it runs
        on a short-delay timer thread, so a synchronous append never
        blocks on title work and never reenters its own publish.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            task = loop.create_task(coro)
            self._in_flight.add(task)
            task.add_done_callback(self._task_done)
            return
        timer = threading.Timer(
            self._DEFER_DELAY_S, lambda: self._run_deferred_in_thread(coro, timer)
        )
        self._timers.add(timer)
        timer.daemon = True
        timer.start()

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._in_flight.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None and not self._disposed:
            logger.warning("session-title: deferred title work failed: %s", error)

    def _run_deferred_in_thread(
        self, coro: Coroutine[Any, Any, None], timer: threading.Timer
    ) -> None:
        try:
            if self._disposed:
                # Disposed while the timer waited: never run the work.
                coro.close()
                return
            asyncio.run(coro)
        except Exception as error:  # noqa: BLE001 - deferred work must not crash its timer thread
            logger.warning("session-title: deferred title work failed: %s", error)
        finally:
            self._timers.discard(timer)
