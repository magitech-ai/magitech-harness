"""Phase 2 acceptance tests (P2-8): the whole session core, end to end.

Each test here proves one item of the Phase 2 acceptance proof against
the public seams only — a stored log file, the persistence backend, the
session service, the title service, and the checkpoint policy — never
against the migration or fold internals those seams stand on. The
fixtures are small hand-built logs modeled on the frozen historical
shapes; no recorded corpus is copied into this file.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from magitech_harness import create_harness
from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.llm.types import ModelChunk, ModelResponse
from magitech_harness.session.events import EventType, ToolCallData
from magitech_harness.session.format import (
    SessionFormatUnsupportedError,
    SessionPersistenceCorruptionError,
)
from magitech_harness.session.messages import (
    create_assistant_message,
    create_system_message,
    create_tool_result_message,
    create_user_message,
)
from magitech_harness.session.persistence_jsonl import JsonlSessionPersistence
from magitech_harness.session.service import SessionService
from magitech_harness.session.store import Session
from magitech_harness.tools.definition import define_tool


def _dump(value: Any) -> str:
    """Canonical one-line JSON for a stored log row."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _header(session_id: str, version: int, **extra: Any) -> dict[str, Any]:
    # The released v0/v1 physical header carries no isSeeded member; v2+
    # does. Mirrors the historical header shapes exactly.
    header: dict[str, Any] = {
        "type": "session",
        "version": version,
        "id": session_id,
        "createdAt": 1000,
        "delegationDepth": 0,
    }
    if version >= 2:
        header["isSeeded"] = False
    header.update(extra)
    return header


def _row(
    type: str,
    seq: int,
    data: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": type,
        "seq": seq,
        "time": 1000 + seq,
        "data": data,
    }
    row.update(extra)
    return row


def _write_log(path: Path, header: dict[str, Any], rows: list[Any]) -> Path:
    """Write one stored log file: a header line, then one row per line."""
    lines = [_dump(header)]
    lines.extend(row if isinstance(row, str) else _dump(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _turn(
    session_id: str, *, version: int = 4, system_text: str = "Be helpful."
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One well-formed turn: system head, two user messages, one reply."""
    header = _header(session_id, version)
    rows = [
        _row("turn/start", 0, {"turn": 1}),
        _row("step/start", 1, {"turn": 1, "step": 1}),
        _row(
            "system/message",
            2,
            {
                "turn": 1,
                "step": 1,
                "message": create_system_message(system_text),
            },
            surfaceOp="append",
        ),
        _row(
            "user/message",
            3,
            create_user_message(
                content=[{"type": "text", "text": "First question"}],
                source={"kind": "user"},
            ),
            surfaceOp="append",
        ),
        _row(
            "user/message",
            4,
            create_user_message(
                content=[{"type": "text", "text": "Second question"}],
                source={"kind": "user"},
            ),
            surfaceOp="append",
        ),
        _row(
            "assistant/message",
            5,
            {
                "turn": 1,
                "step": 1,
                "message": create_assistant_message(
                    content=[{"type": "text", "text": "Here is the answer."}],
                    provider="p",
                    model="m",
                ),
                "stream": [],
            },
            surfaceOp="append",
        ),
        _row("step/end", 6, {"turn": 1, "step": 1}),
        _row("turn/end", 7, {"turn": 1, "reason": {"kind": "done"}}),
    ]
    return header, rows


def _message_shape(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Role plus text of each derived message, in order."""
    shape = []
    for message in messages:
        texts = "".join(
            block.get("text", "")
            for block in message.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        shape.append((message.get("role", ""), texts))
    return shape


# ---------------------------------------------------------------------------
# (a) Round-trip: historical logs open, current logs reload identically.
# ---------------------------------------------------------------------------


def test_acceptance_pre_step_v0_rows_refused_on_load(tmp_path: Path) -> None:
    """A v0 log with surface rows before the first step is refused, not repaired.

    The frozen pre-step shape — user messages ahead of any step — cannot
    acquire a system head without changing chronology, so the load refuses
    with the unsupported-migration text and the file is left untouched.
    """
    file_path = tmp_path / "pre-step-v0.jsonl"
    before = _write_log(
        file_path,
        _header("pre-step-v0", 0, cwd="/work"),
        [
            _row("turn/start", 0, {"turn": 1}),
            _row(
                "user/message",
                1,
                create_user_message(
                    content=[{"type": "text", "text": "A"}],
                    source={"kind": "user"},
                ),
                surfaceOp="append",
            ),
            _row(
                "user/message",
                2,
                create_user_message(
                    content=[{"type": "text", "text": "B"}],
                    source={"kind": "user"},
                ),
                surfaceOp="append",
            ),
            _row("step/start", 3, {"turn": 1, "step": 1}),
            _row("turn/end", 4, {"turn": 1, "reason": {"kind": "done"}}),
        ],
    ).read_bytes()
    with pytest.raises(
        SessionFormatUnsupportedError, match="surface before first step"
    ):
        Session.load(file_path)
    assert file_path.read_bytes() == before


def test_acceptance_pre_step_v0_rows_refused_by_persistence_open(
    tmp_path: Path,
) -> None:
    """The same refusal holds at the backend seam, on read and on write.

    Neither access publishes a successor generation, and the source
    generation is byte-identical afterwards.
    """
    root = tmp_path / "root"
    source = root / "_no-cwd" / "pre-step-v0" / "session.jsonl"
    source.parent.mkdir(parents=True)
    _write_log(
        source,
        _header("pre-step-v0", 0, cwd="/work"),
        [
            _row("turn/start", 0, {"turn": 1}),
            _row(
                "user/message",
                1,
                create_user_message(
                    content=[{"type": "text", "text": "A"}],
                    source={"kind": "user"},
                ),
                surfaceOp="append",
            ),
            _row("step/start", 2, {"turn": 1, "step": 1}),
            _row("turn/end", 3, {"turn": 1, "reason": {"kind": "done"}}),
        ],
    )
    before = source.read_bytes()
    target = root / "_no-cwd" / "pre-step-v0" / "session.v4.jsonl"
    svc = JsonlSessionPersistence(root)
    for access in ("read", "write"):
        with pytest.raises(
            SessionFormatUnsupportedError, match="surface before first step"
        ):
            svc.open("pre-step-v0", access)
    assert source.read_bytes() == before
    assert not target.exists()
    names = {
        name
        for name in (root / "_no-cwd" / "pre-step-v0").iterdir()
        if name.name != "session.lock"
    }
    assert {path.name for path in names} == {"session.jsonl"}


def test_acceptance_valid_v2_chronology_migrates_and_replays(
    tmp_path: Path,
) -> None:
    """A v2 log with valid chronology migrates: system head promoted, replace canonical.

    The promoted system message lands at seq 2, and the replacement
    envelope is rewritten to the canonical startSeq/endSeq keys. The
    derived sequence is the system head plus the replacement message.
    """
    file_path = tmp_path / "valid-v2.jsonl"
    question = create_user_message(
        content=[{"type": "text", "text": "question"}],
        source={"kind": "user"},
    )
    summary = {**question, "id": "summary"}
    summary["content"] = [{"type": "text", "text": "condensed question"}]
    _write_log(
        file_path,
        _header("valid-v2", 2),
        [
            _row("turn/start", 0, {"turn": 1}),
            _row("step/start", 1, {"turn": 1, "step": 1}),
            _row("user/message", 2, question, surfaceOp="append"),
            _row(
                "user/message",
                3,
                summary,
                surfaceOp={"op": "replace", "start": 2, "end": 2},
                sourceEventSeqs=[2],
            ),
            _row("step/end", 4, {"turn": 1, "step": 1}),
            _row("turn/end", 5, {"turn": 1, "reason": {"kind": "completed"}}),
        ],
    )
    loaded = Session.load(file_path)
    assert [event.type for event in loaded.events] == [
        "turn/start",
        "step/start",
        "system/message",
        "user/message",
        "user/message",
        "step/end",
        "turn/end",
    ]
    assert [event.seq for event in loaded.events] == [0, 1, 2, 3, 4, 5, 6]
    replacement = loaded.events[4]
    assert replacement.surface_op == {
        "op": "replace",
        "startSeq": 3,
        "endSeq": 3,
    }
    assert "start" not in replacement.surface_op
    assert "end" not in replacement.surface_op
    # The promoted system head carries an empty prompt, so it is log-only:
    # it exists in the restored log but derives to no message.
    assert loaded.events[2].payload["message"]["content"] == []
    assert _message_shape(loaded.derive_messages()) == [
        ("user", "condensed question"),
    ]


def test_acceptance_harness_written_session_reloads_with_identical_messages(
    tmp_path: Path,
) -> None:
    """A session written through the harness reloads with identical messages.

    The stored file carries a v4 header, and a fresh load derives the
    same role-and-text sequence as the live session.
    """
    session = Session(session_id="written")
    session.append("turn/start", {"turn": 1})
    session.append("step/start", {"turn": 1, "step": 1})
    session.append(
        "system/message",
        {
            "turn": 1,
            "step": 1,
            "message": create_system_message("Be helpful."),
        },
        surface_op="append",
    )
    session.append(
        "user/message",
        create_user_message(
            content=[{"type": "text", "text": "First question"}],
            source={"kind": "user"},
        ),
        surface_op="append",
    )
    session.append(
        "assistant/message",
        {
            "turn": 1,
            "step": 1,
            "message": create_assistant_message(
                content=[{"type": "text", "text": "Here is the answer."}],
                provider="p",
                model="m",
            ),
            "stream": [],
        },
        surface_op="append",
    )
    session.append("step/end", {"turn": 1, "step": 1})
    session.append("turn/end", {"turn": 1, "reason": {"kind": "done"}})
    live_shape = _message_shape(session.derive_messages())

    root = tmp_path / "root"
    svc = JsonlSessionPersistence(root)
    handle = svc.create(session.header)
    handle.append(session.events)
    handle.flush()
    handle.close()

    stored = root / "_no-cwd" / "written" / "session.v4.jsonl"
    first_line = json.loads(stored.read_text(encoding="utf-8").splitlines()[0])
    assert first_line["type"] == "session"
    assert first_line["version"] == 4
    assert first_line["id"] == "written"
    reloaded = Session.load(stored)
    assert _message_shape(reloaded.derive_messages()) == live_shape
    assert _message_shape(reloaded.derive_messages()) == [
        ("system", "Be helpful."),
        ("user", "First question"),
        ("assistant", "Here is the answer."),
    ]


def test_acceptance_v2_and_v4_generations_replay_identical_sequence(
    tmp_path: Path,
) -> None:
    """The same turn stored as v2 and as v4 replays to the same sequence.

    The v2 generation acquires its system head by promotion; the v4
    generation carries it explicitly. Roles and texts agree exactly.
    """
    v2_header, v2_rows = _turn("same-turn", version=2)
    # The v2 generation carries no explicit system head: the promotion
    # supplies it from the request header's system prompt. Drop the
    # explicit system row, insert the prompt-carrying header, and
    # renumber so the source stays dense.
    v2_rows = [row for row in v2_rows if row["type"] != "system/message"]
    v2_rows.insert(
        2,
        _row(
            "request/header",
            2,
            {
                "header": {"system": "Be helpful."},
                "reason": {"kind": "start"},
            },
        ),
    )
    for index, row in enumerate(v2_rows):
        row["seq"] = index
        row["time"] = 1000 + index
    v2_path = tmp_path / "same-v2.jsonl"
    _write_log(v2_path, v2_header, v2_rows)
    v4_header, v4_rows = _turn("same-turn", version=4)
    v4_path = tmp_path / "same-v4.jsonl"
    _write_log(v4_path, v4_header, v4_rows)
    assert _message_shape(Session.load(v2_path).derive_messages()) == (
        _message_shape(Session.load(v4_path).derive_messages())
    )


# ---------------------------------------------------------------------------
# (b) Replay of a recorded session reproduces the identical message sequence.
# ---------------------------------------------------------------------------


def test_acceptance_replay_recorded_turn_reproduces_identical_sequence(
    tmp_path: Path,
) -> None:
    """Two loads of one recorded turn derive the identical message sequence.

    Consecutive same-role user messages stay separate nodes, a failed
    model attempt is preserved in the log but never appears in the
    derived history, and the tool pair replays in order.
    """
    file_path = tmp_path / "recorded-turn.jsonl"
    _write_log(
        file_path,
        _header("recorded-turn", 4),
        [
            _row("turn/start", 0, {"turn": 1}),
            _row("step/start", 1, {"turn": 1, "step": 1}),
            _row(
                "system/message",
                2,
                {
                    "turn": 1,
                    "step": 1,
                    "message": create_system_message("Be helpful."),
                },
                surfaceOp="append",
            ),
            _row(
                "user/message",
                3,
                create_user_message(
                    content=[{"type": "text", "text": "First question"}],
                    source={"kind": "user"},
                ),
                surfaceOp="append",
            ),
            _row(
                "user/message",
                4,
                create_user_message(
                    content=[{"type": "text", "text": "Second question"}],
                    source={"kind": "user"},
                ),
                surfaceOp="append",
            ),
            _row("assistant/attempt", 5, {"turn": 1, "step": 1, "stream": []}),
            _row(
                "assistant/message",
                6,
                {
                    "turn": 1,
                    "step": 1,
                    "message": create_assistant_message(
                        content=[{"type": "text", "text": "Adding now."}],
                        provider="p",
                        model="m",
                    ),
                    "stream": [],
                },
                surfaceOp="append",
            ),
            _row(
                "tool/call",
                7,
                {
                    "turn": 1,
                    "step": 1,
                    "callId": "c1",
                    "name": "add",
                    "arguments": {"a": 1},
                },
            ),
            _row(
                "tool/result",
                8,
                {
                    "turn": 1,
                    "step": 1,
                    "message": create_tool_result_message(
                        call_id="c1",
                        content=[{"type": "text", "text": "3"}],
                    ),
                },
                surfaceOp="append",
            ),
            _row("step/end", 9, {"turn": 1, "step": 1}),
            _row("step/start", 10, {"turn": 1, "step": 2}),
            _row(
                "assistant/message",
                11,
                {
                    "turn": 1,
                    "step": 2,
                    "message": create_assistant_message(
                        content=[{"type": "text", "text": "Done."}],
                        provider="p",
                        model="m",
                    ),
                    "stream": [],
                },
                surfaceOp="append",
            ),
            _row("step/end", 12, {"turn": 1, "step": 2}),
            _row("turn/end", 13, {"turn": 1, "reason": {"kind": "done"}}),
        ],
    )
    first = Session.load(file_path)
    second = Session.load(file_path)
    assert "assistant/attempt" in [event.type for event in first.events]
    assert first.derive_messages() == second.derive_messages()
    assert _message_shape(first.derive_messages()) == [
        ("system", "Be helpful."),
        ("user", "First question"),
        ("user", "Second question"),
        ("assistant", "Adding now."),
        ("tool", "3"),
        ("assistant", "Done."),
    ]


def test_acceptance_compaction_summary_is_real_message_not_synthesized(
    tmp_path: Path,
) -> None:
    """A compaction checkpoint replays through its replacement message only.

    The summary event contributes no message of its own, and no summary
    text is invented at read time: the checkpoint message is the one
    real user message, and the later message stays a separate node.
    """
    file_path = tmp_path / "compacted.jsonl"
    summary_blocks = [{"type": "text", "text": "Initial task: step 1 done."}]
    replacement = create_user_message(
        content=[
            {"type": "text", "text": "Summary of earlier conversation:\n"},
            *summary_blocks,
            {"type": "text", "text": "\nEnd of summary."},
        ],
        source={"kind": "compact-checkpoint", "compactionId": "c1"},
    )
    later = create_user_message(
        content=[{"type": "text", "text": "Now step 2"}],
        source={"kind": "user"},
    )
    _write_log(
        file_path,
        _header("compacted", 4),
        [
            _row("compaction/start", 0, {"compactionId": "c1", "turn": 1}),
            _row(
                "user/message",
                1,
                create_user_message(
                    content=[{"type": "text", "text": "Initial task"}],
                    source={"kind": "user"},
                ),
                surfaceOp="append",
            ),
            _row(
                "assistant/message",
                2,
                {
                    "turn": 1,
                    "step": 1,
                    "message": create_assistant_message(
                        content=[{"type": "text", "text": "Step 1 done"}],
                        provider="p",
                        model="m",
                    ),
                    "stream": [],
                },
                surfaceOp="append",
            ),
            _row(
                "compaction/summary",
                3,
                {
                    "compactionId": "c1",
                    "summary": summary_blocks,
                    "shadowedRange": {"start": 1, "end": 2},
                    "shadowedSeqs": [1, 2],
                    "shadowedTokenCount": 10,
                    "provider": "p",
                    "model": "m",
                },
            ),
            _row(
                "user/message",
                4,
                replacement,
                surfaceOp={"op": "replace", "startSeq": 1, "endSeq": 2},
                sourceEventSeqs=[0, 3, 1, 2],
            ),
            _row("user/message", 5, later, surfaceOp="append"),
            _row(
                "assistant/message",
                6,
                {
                    "turn": 1,
                    "step": 1,
                    "message": create_assistant_message(
                        content=[{"type": "text", "text": "Step 2 done"}],
                        provider="p",
                        model="m",
                    ),
                    "stream": [],
                },
                surfaceOp="append",
            ),
        ],
    )
    messages = Session.load(file_path).derive_messages()
    assert len(messages) == 3
    assert messages[0] == replacement
    assert messages[1] == later
    assert messages[2]["role"] == "assistant"
    assert not any(
        "Initial task: step 1 done." in block.get("text", "")
        for message in messages[1:]
        for block in message.get("content", [])
        if isinstance(block, dict)
    )


# ---------------------------------------------------------------------------
# (c) Rename pins against auto-title.
# ---------------------------------------------------------------------------


class _FakeTitleProvider:
    """Scriptable title provider recording every request."""

    def __init__(self, *, title: str = "Provider Title") -> None:
        self.id = "fake-provider"
        self.automatic = "first-prompt"
        self.requests: list[Any] = []
        self._title = title
        self.gate: asyncio.Event | None = None

    async def generate(self, request: Any) -> dict[str, Any]:
        self.requests.append(request)
        if self.gate is not None and len(self.requests) == 1:
            await self.gate.wait()
        return {
            "title": self._title,
            "messageSeqs": [message["seq"] for message in request.messages],
        }


def _log_user(service: SessionService, session: Session, text: str) -> None:
    service.log_event(
        session,
        EventType.USER_MESSAGE,
        {
            "content": [{"type": "text", "text": text}],
            "source": {"kind": "user"},
        },
    )


def _log_header(service: SessionService, session: Session) -> None:
    service.log_event(
        session,
        EventType.REQUEST_HEADER,
        {
            "header": {"config": {"provider": "prov", "model": "mod"}},
            "reason": "initial",
        },
    )


async def test_acceptance_rename_pins_against_auto_title(tmp_path: Path) -> None:
    """A rename supersedes in-flight automatic titling and survives restart.

    The renamed title stands through subsequent turns with no further
    provider call, and a fresh service loading the same log still reads
    the pinned title — the title lives in the log, not in memory.
    """
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    service = ctx.sessions
    title = ctx.sessionTitle
    session = service.create("pinned")
    provider = _FakeTitleProvider()
    provider.gate = asyncio.Event()
    title.register(provider)
    _log_user(service, session, "Explain quantum computing simply")
    _log_header(service, session)
    await asyncio.sleep(0)
    assert len(provider.requests) == 1
    title.rename(session, "My Pinned Title")
    provider.gate.set()
    await title.drain()
    assert title.get(session)["title"] == "My Pinned Title"
    assert title.get(session)["source"] == {"kind": "user"}
    _log_user(service, session, "And now explain relativity")
    _log_header(service, session)
    await title.drain()
    _log_user(service, session, "And now explain black holes")
    _log_header(service, session)
    await title.drain()
    assert len(provider.requests) == 1
    assert title.get(session)["title"] == "My Pinned Title"
    title_events = [event for event in session.events if event.type == "session/title"]
    assert len(title_events) == 2  # fallback + rename

    await service.flush()
    assert service.close("pinned") is True
    ctx2 = create_harness(storage_dir=tmp_path)
    await ctx2.flush()
    reloaded_service = ctx2.sessions
    loaded = reloaded_service.load("pinned")
    assert loaded is not None
    assert ctx2.sessionTitle.get(loaded)["title"] == "My Pinned Title"
    assert [
        event.payload["title"]
        for event in loaded.events
        if event.type == "session/title"
    ] == ["Explain quantum computing simply", "My Pinned Title"]


# ---------------------------------------------------------------------------
# (d) Checkpoint rejection prevents adapter dispatch.
# ---------------------------------------------------------------------------


class _RecordingProvider:
    """Provider that records adapter invocations in call order."""

    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.calls = 0

    async def generate(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ModelResponse:
        return ModelResponse(text="hello")

    async def stream(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[ModelChunk]:
        self.calls += 1
        self.order.append("adapter")
        yield ModelChunk(text="hello")


async def _drain(stream: Any) -> list[Any]:
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    return chunks


async def test_acceptance_failing_flush_blocks_llm_and_tool_dispatch(
    tmp_path: Path,
) -> None:
    """A rejected checkpoint blocks the model adapter and the tool body alike.

    With a failing flush, the model stream raises before any adapter
    invocation, and a top-level tool call returns an error result with
    its body never run. A healthy flush lets both proceed.
    """
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    order: list[str] = []
    provider = _RecordingProvider(order)
    ctx.llm.register_provider("rec", provider)
    ctx.llm.set_active_provider("rec")
    session = ctx.sessions.create("dispatch-gate")
    ran = False

    async def failing_flush(flushed: Any) -> None:
        raise RuntimeError("disk unavailable")

    ctx.on("session/flush", failing_flush)

    def body(args: dict[str, Any]) -> str:
        nonlocal ran
        ran = True
        return "done"

    ctx.tools.register(
        define_tool(name="write", description="side effect", execute=body)
    )

    with pytest.raises(RuntimeError, match="disk unavailable"):
        await _drain(ctx.llm.stream(messages=[], session_id=session.session_id))
    assert order == []
    assert provider.calls == 0

    result = await ctx.tools.execute(
        ToolCallData(call_id="write-1", name="write", arguments={}),
        session=session,
        signal=AbortSignal(),
    )
    assert result.is_error
    assert "disk unavailable" in result.content
    assert ran is False


# ---------------------------------------------------------------------------
# (e) Corrupt / unknown-version files are refused, not silently repaired.
# ---------------------------------------------------------------------------


def test_acceptance_corrupt_row_refused_not_repaired(tmp_path: Path) -> None:
    """A torn row refuses the load, and the file is left byte-identical."""
    file_path = tmp_path / "torn.jsonl"
    header, rows = _turn("torn")
    _write_log(file_path, header, rows)
    before = file_path.read_bytes()
    torn = before.decode("utf-8").splitlines()
    torn[3] = '{"type": "user/message", "seq": 3, "time": 1003, "data": {"text": "tor'
    file_path.write_text("\n".join(torn) + "\n", encoding="utf-8")
    expected = file_path.read_bytes()
    with pytest.raises(SessionPersistenceCorruptionError, match="corrupt session row"):
        Session.load(file_path)
    assert file_path.read_bytes() == expected


def test_acceptance_unknown_version_refused_without_decoding_rows(
    tmp_path: Path,
) -> None:
    """A newer-version header is refused with the upgrade text, rows unread.

    The row after the header is not JSON at all: had the loader decoded
    rows before classifying the header, it would have failed differently.
    """
    file_path = tmp_path / "future.jsonl"
    file_path.write_text(
        _dump(_header("future", 9))
        + "\n"
        + "this is not json at all\n"
        + _dump(_row("user/message", 1, {"text": "x"}))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SessionFormatUnsupportedError, match="upgrade the harness"):
        Session.load(file_path)


def test_acceptance_unknown_non_ignorable_type_refused_at_load(
    tmp_path: Path,
) -> None:
    """An unknown non-ignorable type refuses the load; an ignorable one is carried.

    The ignorable record stays in the restored log, opaque, and projects
    to no message.
    """
    refused = tmp_path / "unknown.jsonl"
    header, rows = _turn("unknown")
    rows.insert(
        3,
        _row(
            "widget/frobnicate",
            3,
            {"note": "unrecognized"},
        ),
    )
    for index, row in enumerate(rows[4:], start=4):
        row["seq"] = index
        row["time"] = 1000 + index
    _write_log(refused, header, rows)
    with pytest.raises(SessionFormatUnsupportedError, match="unknown event type"):
        Session.load(refused)

    carried = tmp_path / "carried.jsonl"
    header, rows = _turn("carried")
    rows.insert(
        3,
        _row("widget/frobnicate", 3, {"note": "unrecognized"}, ignorable=True),
    )
    for index, row in enumerate(rows[4:], start=4):
        row["seq"] = index
        row["time"] = 1000 + index
    _write_log(carried, header, rows)
    loaded = Session.load(carried)
    assert "widget/frobnicate" in [event.type for event in loaded.events]
    assert _message_shape(loaded.derive_messages()) == [
        ("system", "Be helpful."),
        ("user", "First question"),
        ("user", "Second question"),
        ("assistant", "Here is the answer."),
    ]
