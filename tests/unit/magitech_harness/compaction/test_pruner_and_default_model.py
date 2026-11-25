"""Tests for the tool-result pruner and the default agent model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from animas import Context

from magitech_harness import create_harness
from magitech_harness.agent.default_model import AgentDefaultModel
from magitech_harness.compaction.pruner import (
    DEFAULT_HEAD_CHARS,
    DEFAULT_TAIL_CHARS,
    DEFAULT_THRESHOLD_CHARS,
    PRUNE_MARKER,
    PruneConfigError,
    ToolResultPruner,
    code_point_length,
    resolve_config,
)
from magitech_harness.llm.service import MockLLMProvider

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_resolve_config_defaults() -> None:
    assert resolve_config() == {
        "thresholdChars": DEFAULT_THRESHOLD_CHARS,
        "headChars": DEFAULT_HEAD_CHARS,
        "tailChars": DEFAULT_TAIL_CHARS,
    }


def test_resolve_config_rejects_unknown_key() -> None:
    with pytest.raises(PruneConfigError, match="unknown key"):
        resolve_config({"nope": 1})


@pytest.mark.parametrize(
    "config,match",
    [
        ({"thresholdChars": 0}, "thresholdChars .* positive integer"),
        ({"thresholdChars": True}, "thresholdChars .* positive integer"),
        ({"headChars": -1}, "headChars .* non-negative integer"),
        ({"tailChars": -1}, "tailChars .* non-negative integer"),
        (
            {"thresholdChars": 10, "headChars": 8, "tailChars": 8},
            "must be at most thresholdChars",
        ),
    ],
)
def test_resolve_config_rejects_bad_budgets(config: dict[str, Any], match: str) -> None:
    with pytest.raises(PruneConfigError, match=match):
        resolve_config(config)


def test_code_point_length_counts_points() -> None:
    assert code_point_length("abc") == 3
    assert code_point_length("😀") == 1


# ---------------------------------------------------------------------------
# Content pruning
# ---------------------------------------------------------------------------


def _pruner(**config: Any) -> ToolResultPruner:
    return ToolResultPruner(Context(), config)


def test_prune_content_under_budget_returns_none() -> None:
    pruner = _pruner(thresholdChars=100, headChars=10, tailChars=10)
    assert pruner.prune_content([{"type": "text", "text": "short"}]) is None


def test_prune_content_keeps_head_and_tail() -> None:
    pruner = _pruner(thresholdChars=60, headChars=8, tailChars=8)
    text = "0123456789" * 50
    pruned = pruner.prune_content([{"type": "text", "text": text}])
    assert pruned is not None
    assert len(pruned) == 1
    body = pruned[0]["text"]
    assert body.startswith("01234")
    assert body.endswith("56789")
    assert PRUNE_MARKER in body
    assert pruner.measure_content(pruned) <= 60


def test_prune_content_preserves_non_text_blocks_in_order() -> None:
    pruner = _pruner(thresholdChars=60, headChars=8, tailChars=8)
    blocks = [
        {"type": "image", "data": "x"},
        {"type": "text", "text": "0123456789" * 50},
        {"type": "image", "data": "y"},
    ]
    pruned = pruner.prune_content(blocks)
    assert pruned is not None
    assert pruned[0] == {"type": "image", "data": "x"}
    assert pruned[-1] == {"type": "image", "data": "y"}


def test_prune_content_inserts_one_marker_across_blocks() -> None:
    pruner = _pruner(thresholdChars=60, headChars=8, tailChars=8)
    blocks = [
        {"type": "text", "text": "a" * 200},
        {"type": "text", "text": "b" * 200},
    ]
    pruned = pruner.prune_content(blocks)
    assert pruned is not None
    joined = "".join(str(block.get("text", "")) for block in pruned)
    assert joined.count(PRUNE_MARKER) == 1


def test_prune_content_handles_non_list() -> None:
    pruner = _pruner()
    assert pruner.prune_content("not blocks") is None
    assert pruner.measure_content(None) == 0


# ---------------------------------------------------------------------------
# Session pruning
# ---------------------------------------------------------------------------


def _message(text: str, call_id: str = "c1") -> dict[str, Any]:
    return {
        "id": f"t-{call_id}",
        "role": "tool",
        "toolCallId": call_id,
        "isError": False,
        "source": {"kind": "tool", "callId": call_id},
        "content": [{"type": "text", "text": text}],
    }


def _tool_result(session: Any, text: str, call_id: str = "c1") -> Any:
    return session.append(
        "tool/result",
        {"turn": 1, "step": 1, "message": _message(text, call_id)},
        surface_op="append",
    )


def test_prune_session_rewrites_only_over_budget_results(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    session = ctx.sessions.create("prune")
    small = _tool_result(session, "small", "c-small")
    big = _tool_result(session, "x" * 500, "c-big")

    pruner = ToolResultPruner(
        ctx, {"thresholdChars": 100, "headChars": 20, "tailChars": 20}
    )
    result = pruner.prune_session(session)
    assert result["pruned"] is True
    assert len(result["entries"]) == 1
    entry = result["entries"][0]
    assert entry["originalSeq"] == big.seq
    assert entry["callId"] == "c-big"
    assert entry["charsBefore"] == 500
    assert entry["charsAfter"] < 500
    assert result["charsRemoved"] == 500 - entry["charsAfter"]

    # The small result keeps its node; the big one is replaced by a new node.
    assert small.seq in session.surface_nodes()
    assert big.seq not in session.surface_nodes()
    assert entry["replacementSeq"] in session.surface_nodes()

    # A shadow-price event immediately precedes each replacement.
    prune_event = session.event_at(entry["replacementSeq"] - 1)
    assert prune_event is not None and prune_event.type == "compaction/prune"
    assert prune_event.payload["shadowedSeqs"] == [big.seq]
    assert prune_event.payload["shadowedTokenCount"] >= 0

    replacement = session.event_at(entry["replacementSeq"])
    assert replacement is not None
    text = replacement.payload["message"]["content"][0]["text"]
    assert PRUNE_MARKER in text


def test_prune_session_no_candidates(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    session = ctx.sessions.create("prune-empty")
    _tool_result(session, "small")
    pruner = ToolResultPruner(
        ctx, {"thresholdChars": 100, "headChars": 20, "tailChars": 20}
    )
    result = pruner.prune_session(session)
    assert result == {"pruned": False, "entries": [], "charsRemoved": 0}


def test_prune_session_skips_malformed_events(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    session = ctx.sessions.create("prune-malformed")
    session.append("turn/start", {"turn": 1})
    pruner = ToolResultPruner(
        ctx, {"thresholdChars": 100, "headChars": 20, "tailChars": 20}
    )
    result = pruner.prune_session(session)
    assert result["pruned"] is False


def test_pruner_is_registered_with_compaction(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    assert "tool-result" in ctx.compaction.list_pruners()


def test_call_id_falls_back_to_tool_call_id() -> None:
    from magitech_harness.compaction.pruner import _call_id

    assert _call_id({"toolCallId": "c-fallback"}) == "c-fallback"
    assert _call_id({"toolCallId": 1}) is None
    assert _call_id({}) is None


def test_estimate_message_fallback_without_meter() -> None:
    from magitech_harness.compaction.pruner import _estimate_message
    from magitech_harness.session.events import SessionEvent

    ctx = Context()
    message = _message("abcd" * 10)
    assert _estimate_message(ctx, message) == 10
    # A meter service that lacks the method also falls back.
    ctx.reflect.provide("tokenMeter", object())
    assert _estimate_message(ctx, message) == 10
    event = SessionEvent(
        id="x", session_id="s", type="tool/result", timestamp=0.0, payload={}
    )
    assert event.type == "tool/result"


# ---------------------------------------------------------------------------
# Default agent model
# ---------------------------------------------------------------------------


def test_default_model_unset_returns_none() -> None:
    service = AgentDefaultModel(Context(), {"provider": "", "model": ""})
    assert service.get() is None
    assert service.apply_to({"provider": "a", "model": "b"}) == {
        "provider": "a",
        "model": "b",
    }


def test_default_model_merges_only_missing_keys() -> None:
    service = AgentDefaultModel(Context(), {"provider": "p", "model": "m"})
    assert service.get() == {"provider": "p", "model": "m"}
    assert service.apply_to(None) == {"provider": "p", "model": "m"}
    assert service.apply_to({"provider": "other"}) == {
        "provider": "other",
        "model": "m",
    }


def test_default_model_set_and_type_checks() -> None:
    service = AgentDefaultModel(Context(), {})
    service.set("p", "m")
    assert service.get() == {"provider": "p", "model": "m"}
    with pytest.raises(TypeError, match="provider must be a string"):
        AgentDefaultModel(Context(), {"provider": 1})


@pytest.mark.asyncio
async def test_agents_create_uses_default_model(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    ctx.llm.register_provider("mock", MockLLMProvider())
    ctx.agentDefaultModel.set("mock", "mock")

    handle = await ctx.agents.create({})
    agent = handle["agent"]
    assert agent.options["provider"] == "mock"
    assert agent.options["model"] == "mock"

    # An explicit selection is never overwritten.
    explicit = await ctx.agents.create(
        {"agentOptions": {"provider": "mock", "model": "explicit"}}
    )
    assert explicit["agent"].options["model"] == "explicit"


@pytest.mark.asyncio
async def test_agents_create_without_default_keeps_error(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    ctx.agentDefaultModel.set("", "")
    handle = await ctx.agents.create({})
    agent = handle["agent"]
    assert "provider" not in agent.options
    assert json.dumps(agent.options) in ("{}", "")


def test_prune_content_empty_text_block_is_dropped() -> None:
    pruner = _pruner(thresholdChars=60, headChars=8, tailChars=8)
    blocks = [
        {"type": "text", "text": "z" * 200},
        {"type": "text", "text": ""},
    ]
    pruned = pruner.prune_content(blocks)
    assert pruned is not None
    assert all(block.get("text") for block in pruned)


def test_prune_content_marker_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    pruner = _pruner(thresholdChars=60, headChars=8, tailChars=8)
    monkeypatch.setattr(pruner, "measure_content", lambda _blocks: 1000)
    with pytest.raises(ValueError, match="failed to locate the removed text span"):
        pruner.prune_content([{"type": "text", "text": "short"}])


def test_prune_content_size_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    pruner = _pruner(thresholdChars=60, headChars=8, tailChars=8)
    values = iter([1000, 5000])
    monkeypatch.setattr(pruner, "measure_content", lambda _blocks: next(values))
    with pytest.raises(ValueError, match="replacement must be smaller"):
        pruner.prune_content([{"type": "text", "text": "x" * 20}])


class _FakeEvent:
    def __init__(self, event_type: str, payload: dict[str, Any]) -> None:
        self.type = event_type
        self.payload = payload


class _FakeSession:
    def __init__(self, events: list[_FakeEvent]) -> None:
        self._events = events

    def surface_nodes(self) -> list[int]:
        return list(range(len(self._events)))

    def event_at(self, seq: int) -> _FakeEvent:
        return self._events[seq]

    def append(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("no append expected")


def test_prune_session_skips_non_tool_surface_nodes(tmp_path: Path) -> None:
    from magitech_harness.llm.message import create_user_message

    ctx = create_harness(storage_dir=tmp_path)
    session = ctx.sessions.create("prune-user")
    session.append(
        "user/message",
        create_user_message(content=[{"type": "text", "text": "hi"}]),
        surface_op="append",
    )
    pruner = ToolResultPruner(
        ctx, {"thresholdChars": 100, "headChars": 20, "tailChars": 20}
    )
    assert pruner.prune_session(session)["pruned"] is False


def test_prune_session_skips_non_mapping_message(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    pruner = ToolResultPruner(
        ctx, {"thresholdChars": 100, "headChars": 20, "tailChars": 20}
    )
    fake = _FakeSession([_FakeEvent("tool/result", {"message": "nope"})])
    assert pruner.prune_session(fake)["pruned"] is False


def test_call_id_falls_back_when_source_call_id_invalid() -> None:
    from magitech_harness.compaction.pruner import _call_id

    assert _call_id({"source": {"callId": 1}, "toolCallId": "x"}) == "x"


def test_estimate_message_uses_meter_when_numeric() -> None:
    from magitech_harness.compaction.pruner import _estimate_message

    class _Meter:
        def estimate_message(self, _message: Any) -> int:
            return 7

    ctx = Context()
    ctx.reflect.provide("tokenMeter", _Meter())
    assert _estimate_message(ctx, _message("abc")) == 7


def test_estimate_message_falls_back_on_non_numeric_meter() -> None:
    from magitech_harness.compaction.pruner import _estimate_message

    class _Meter:
        def estimate_message(self, _message: Any) -> Any:
            return "nope"

    ctx = Context()
    ctx.reflect.provide("tokenMeter", _Meter())
    assert _estimate_message(ctx, _message("abcd")) == 1


@pytest.mark.asyncio
async def test_compaction_prune_runs_registered_pruner(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    session = ctx.sessions.create("prune-via-service")
    _tool_result(session, "x" * 900, "c-service")
    ctx.toolResultPruner.config.update(
        {"thresholdChars": 100, "headChars": 20, "tailChars": 20}
    )
    result = await ctx.compaction.prune(session)
    assert result["pruned"] is True


# ---------------------------------------------------------------------------
# User questions
# ---------------------------------------------------------------------------


def test_validate_questions_rejects_bad_shapes() -> None:
    from magitech_harness.context.user_questions import (
        UserQuestionError,
        _validate_questions,
    )

    with pytest.raises(UserQuestionError, match="non-empty array"):
        _validate_questions(None)
    with pytest.raises(UserQuestionError, match="must be an object"):
        _validate_questions(["nope"])
    base = {"id": "q1", "question": "Pick one"}
    for field, value, match in [
        ("id", "", "id must be a non-empty string"),
        ("question", "", "question must be a non-empty string"),
        ("header", 1, "header must be a string"),
        ("options", [], "options must be a non-empty array"),
    ]:
        with pytest.raises(UserQuestionError, match=match):
            _validate_questions([{**base, field: value}])
    with pytest.raises(UserQuestionError, match="duplicates"):
        _validate_questions([base, base])
    with pytest.raises(UserQuestionError, match="label must be a non-empty string"):
        _validate_questions([{**base, "options": [{"label": ""}]}])
    with pytest.raises(UserQuestionError, match="description must be a string"):
        _validate_questions([{**base, "options": [{"label": "a", "description": 1}]}])


def test_validate_questions_normalizes() -> None:
    from magitech_harness.context.user_questions import _validate_questions

    result = _validate_questions(
        [
            {
                "id": "q1",
                "question": "Pick",
                "header": "Confirm",
                "multi_select": True,
                "options": [{"label": "Yes", "description": "do it"}],
            }
        ]
    )
    assert result == [
        {
            "id": "q1",
            "question": "Pick",
            "header": "Confirm",
            "multiSelect": True,
            "options": [{"label": "Yes", "description": "do it"}],
        }
    ]


@pytest.mark.asyncio
async def test_user_questions_ask_with_answerer(tmp_path: Path) -> None:
    from magitech_harness.context.user_questions import Answer

    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    called: list[Any] = []

    async def answerer(pending: Any, _signal: Any) -> list[Answer]:
        called.append(pending)
        return [Answer(id="q1", selected=["Yes"])]

    ctx.userQuestions.register_answerer(answerer)
    handle = await ctx.agents.create({})
    answers = await ctx.userQuestions.ask(
        [{"id": "q1", "question": "Pick", "options": [{"label": "Yes"}]}],
        agent=handle["agent"],
    )
    assert answers[0].to_dict() == {"id": "q1", "selected": ["Yes"]}
    assert called and called[0].questions[0]["id"] == "q1"
    questions = [
        event
        for event in handle["agent"].session.events
        if event.type == "user/question"
    ]
    assert len(questions) == 1
    assert questions[0].payload["callId"] == called[0].call_id


@pytest.mark.asyncio
async def test_user_questions_without_answerer_fails(tmp_path: Path) -> None:
    from magitech_harness.context.user_questions import UserQuestionError

    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    with pytest.raises(UserQuestionError, match="no user-question answerer"):
        await ctx.userQuestions.ask([{"id": "q1", "question": "Pick"}])


@pytest.mark.asyncio
async def test_ask_user_question_tool_returns_answers(tmp_path: Path) -> None:
    import asyncio

    from magitech_harness.context.user_questions import Answer
    from magitech_harness.session.events import ToolCallData

    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()

    def answerer(pending: Any, _signal: Any) -> list[Answer]:
        return [Answer(id="q1", selected=["Yes"], custom="because")]

    ctx.userQuestions.register_answerer(answerer)
    session = ctx.sessions.create("ask-tool")
    result = await ctx.tools.execute(
        ToolCallData(
            call_id="ask-1",
            name="ask_user_question",
            arguments={"questions": [{"id": "q1", "question": "Pick"}]},
        ),
        session=session,
        signal=asyncio.Event().is_set
        and __import__(
            "magitech_harness.agent.cancellation", fromlist=["AbortSignal"]
        ).AbortSignal(),
    )
    assert result.is_error is False
    assert result.content["answers"] == [
        {"id": "q1", "selected": ["Yes"], "custom": "because"}
    ]


def test_validate_answers_rejects_bad_values() -> None:
    from magitech_harness.context.user_questions import (
        UserQuestionError,
        UserQuestionService,
    )

    questions = [{"id": "q1"}]
    for value, match in [
        ("nope", "must return a list"),
        ([{"id": "other"}], "does not match a question"),
        ([{"id": "q1"}, {"id": "q1"}], "duplicate answer"),
        ([{"id": "q1", "selected": [1]}], "selected must be an array of strings"),
        ([{"id": "q1", "custom": 1}], "custom must be a string"),
    ]:
        with pytest.raises(UserQuestionError, match=match):
            UserQuestionService._validate_answers(value, questions)


@pytest.mark.asyncio
async def test_answerer_fault_becomes_structured_error(tmp_path: Path) -> None:
    from magitech_harness.context.user_questions import UserQuestionError

    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()

    def bad_answerer(_pending: Any, _signal: Any) -> Any:
        raise RuntimeError("ui broke")

    ctx.userQuestions.register_answerer(bad_answerer)
    with pytest.raises(UserQuestionError, match="ui broke"):
        await ctx.userQuestions.ask([{"id": "q1", "question": "Pick"}])


def test_answerer_disposer_removes_it() -> None:
    from animas import Context

    from magitech_harness.context.user_questions import UserQuestionService

    service = UserQuestionService(Context())
    answerer = lambda _p, _s: []
    dispose = service.register_answerer(answerer)
    assert answerer in service._answerers
    dispose()
    assert answerer not in service._answerers
    # A second disposal is a no-op (the branch when the answerer is absent).
    dispose()
    assert answerer not in service._answerers


@pytest.mark.asyncio
async def test_user_questions_cancel_reraises(tmp_path: Path) -> None:
    import asyncio

    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()

    def cancelling(_pending: Any, _signal: Any) -> Any:
        raise asyncio.CancelledError()

    ctx.userQuestions.register_answerer(cancelling)
    with pytest.raises(asyncio.CancelledError):
        await ctx.userQuestions.ask([{"id": "q1", "question": "Pick"}])


def test_apply_creates_service_when_absent() -> None:
    from animas import Context

    from magitech_harness.context.user_questions import (
        UserQuestionService,
        apply,
    )
    from magitech_harness.tools.service import ToolsService

    ctx = Context()
    ToolsService(ctx)
    apply(ctx)
    assert isinstance(ctx.userQuestions, UserQuestionService)


@pytest.mark.asyncio
async def test_user_questions_aborted_signal_refuses_before_logging(
    tmp_path: Path,
) -> None:
    from magitech_harness.agent.cancellation import AbortSignal
    from magitech_harness.context.user_questions import UserQuestionError

    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    handle = await ctx.agents.create({})
    agent = handle["agent"]
    signal = AbortSignal()
    signal.abort("stop")
    with pytest.raises(UserQuestionError, match="aborted before the user answered"):
        await ctx.userQuestions.ask(
            [{"id": "q1", "question": "Pick"}], agent=agent, signal=signal
        )
    assert not any(event.type == "user/question" for event in agent.session.events)


@pytest.mark.asyncio
async def test_user_questions_abort_after_answer_raises(tmp_path: Path) -> None:
    from magitech_harness.agent.cancellation import AbortSignal
    from magitech_harness.context.user_questions import (
        Answer,
        UserQuestionError,
    )

    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    signal = AbortSignal()

    def answerer(_pending: Any, _signal: Any) -> list[Answer]:
        signal.abort("late")
        return [Answer(id="q1", selected=[])]

    ctx.userQuestions.register_answerer(answerer)
    with pytest.raises(UserQuestionError, match="aborted before the user answered"):
        await ctx.userQuestions.ask([{"id": "q1", "question": "Pick"}], signal=signal)


@pytest.mark.asyncio
async def test_user_questions_answerer_raises_structured_error(tmp_path: Path) -> None:
    from magitech_harness.context.user_questions import UserQuestionError

    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()

    def answerer(_pending: Any, _signal: Any) -> Any:
        raise UserQuestionError("declined", "ASK_DECLINED")

    ctx.userQuestions.register_answerer(answerer)
    with pytest.raises(UserQuestionError, match="declined"):
        await ctx.userQuestions.ask([{"id": "q1", "question": "Pick"}])


@pytest.mark.asyncio
async def test_user_questions_already_settled_future_is_not_cancelled(
    tmp_path: Path,
) -> None:
    from magitech_harness.context.user_questions import Answer

    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()

    def answerer(pending: Any, _signal: Any) -> list[Answer]:
        pending.future.set_result([Answer(id="q1", selected=[])])
        return [Answer(id="q1", selected=[])]

    ctx.userQuestions.register_answerer(answerer)
    answers = await ctx.userQuestions.ask([{"id": "q1", "question": "Pick"}])
    assert answers[0].id == "q1"
