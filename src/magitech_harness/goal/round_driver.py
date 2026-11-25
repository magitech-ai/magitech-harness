"""Same-session goal-round continuation driver.

Ports ``packages/goal/goal-round-driver/src/{index,prompt}.ts``. When an agent
with an armed, active goal becomes quiescent, the driver reserves exactly one
next round through ``agent.followup`` with a ``goal`` message source. The goal
projection counts admitted rounds, so the loop is durable across resume.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from animas import Context, EventMode, Service

from magitech_harness.goal.types import GoalView
from magitech_harness.llm.message import create_user_message

ROUND_LIMIT_CODE = "round-limit"
PROMPT_REJECTED_CODE = "prompt-rejected"
QUEUE_FAILED_CODE = "queue-failed"


def render_goal_round_prompt(goal: GoalView, round: int) -> list[dict[str, Any]]:
    """Render the complete model-visible instruction for one goal round."""
    import json

    text = (
        "<goal_round>\n"
        f"Objective: {json.dumps(goal.objective, ensure_ascii=False)}\n"
        f"Round: {round}/{goal.max_goal_rounds}\n\n"
        "Continue working toward the objective in this same session. Treat the current "
        "workspace, tool results, and durable session state as authoritative; inspect them "
        "instead of assuming earlier narration is still current. Make concrete progress and "
        "verify the result. Before claiming completion, gather evidence that the whole "
        "objective is achieved, read the current goal, and mark it complete. If work "
        "remains, leave the goal active for the next round. Follow the configured goal-tool "
        "policy before reporting a blocker.\n"
        "</goal_round>"
    )
    return [{"type": "text", "text": text}]


@dataclass
class _Round:
    goal_id: str
    revision: int
    round: int
    message_id: str
    content: list[dict[str, Any]] = field(default_factory=list)
    phase: str = "queued"
    cancelled: bool = False
    stale: bool = False


@dataclass
class _State:
    agent: Any = None
    attempt: _Round | None = None
    competing_queued: bool = False
    needs_checkpoint: bool = False
    requested: bool = False
    run: Any = None
    stopping: bool = False


class GoalRoundDriver(Service):
    """Serialized, race-fenced automatic goal continuation."""

    inject: ClassVar[list[str]] = ["agents", "goals", "sessions"]

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, "goalRoundDriver")
        del config
        self._states: dict[str, _State] = {}

    def _state(self, agent: Any) -> _State:
        identifier = str(getattr(agent, "id", None) or id(agent))
        state = self._states.get(identifier)
        if state is None:
            state = _State(agent=agent)
            self._states[identifier] = state
        return state

    def _current_goal(self, agent: Any) -> GoalView | None:
        registry = self.ctx.get("agents", None, strict=False)
        if (
            registry is not None
            and registry.get(getattr(agent, "id", None)) is not agent
        ):
            return None
        goals = self.ctx.get("goals", None, strict=False)
        if goals is None:
            return None
        from typing import cast

        return cast(GoalView | None, goals.get(agent))

    @staticmethod
    def _drivable(goal: GoalView | None) -> bool:
        return (
            goal is not None and goal.phase == "active" and goal.activation == "armed"
        )

    def _ready(self, agent: Any, state: _State) -> bool:
        registry = self.ctx.get("agents", None, strict=False)
        return (
            not state.stopping
            and (registry is None or registry.get(getattr(agent, "id", None)) is agent)
            and getattr(agent, "status", None) == "idle"
            and not state.competing_queued
        )

    def _disarm(self, agent: Any) -> None:
        goals = self.ctx.get("goals", None, strict=False)
        if goals is None:
            return
        try:
            if self._drivable(self._current_goal(agent)):
                goals.disarm(agent)
        except Exception:  # noqa: BLE001, S110 -- a disarm failure must not break the listener
            pass

    def _valid_reservation(
        self,
        state: _State,
        content: list[Any],
        source: Mapping[str, Any],
        goal: GoalView | None,
    ) -> bool:
        """Fail closed unless the queued prompt still owns the exact live revision."""
        attempt = state.attempt
        return (
            not state.stopping
            and attempt is not None
            and attempt.phase == "claimed"
            and not attempt.stale
            and self._same_queued(content, source, attempt)
            and goal is not None
            and goal.id == source.get("goalId")
            and goal.revision == source.get("revision")
            and self._drivable(goal)
            and source.get("round") == goal.rounds_started + 1
        )

    def _restore_other_claimed(
        self, agent: Any, messages: list[Any], message_id: str
    ) -> None:
        inbox = getattr(agent, "inbox", None)
        if inbox is None:
            return
        retained = [
            message
            for message in messages
            if message.get("id") != message_id
            and not (
                isinstance(message.get("source"), Mapping)
                and message["source"].get("kind") == "goal"
                and message["source"].get("round") == 0
            )
        ]
        prepend = getattr(inbox, "prepend", None)
        if not callable(prepend):
            return
        for message in reversed(retained):
            prepend("next-step", message)

    async def drive(self, agent: Any, state: _State) -> None:
        """Process one quiescent wake-up, reserving at most one next round."""
        if not self._ready(agent, state):
            return
        if state.needs_checkpoint:
            state.needs_checkpoint = False
            sessions = self.ctx.get("sessions", None, strict=False)
            if sessions is not None:
                flushed = sessions.flush(agent.session)
                if hasattr(flushed, "__await__"):
                    await flushed
            if not self._ready(agent, state) or state.needs_checkpoint:
                return
        attempt = state.attempt
        if attempt is not None:
            state.attempt = None
            state.needs_checkpoint = True
            state.requested = True
            return
        goal = self._current_goal(agent)
        if not self._drivable(goal):
            return
        subagents = self.ctx.get("subagents", None, strict=False)
        wait = getattr(subagents, "wait_for_tree_work", None)
        if callable(wait):
            waiting = wait(agent)
            if hasattr(waiting, "__await__"):
                await waiting
            if not self._ready(agent, state) or state.needs_checkpoint:
                return
        assert goal is not None
        if goal.rounds_started >= goal.max_goal_rounds:
            self.ctx.goals.block(
                agent,
                {"id": goal.id, "revision": goal.revision},
                {
                    "code": ROUND_LIMIT_CODE,
                    "message": (
                        f"Goal reached its configured limit of "
                        f"{goal.max_goal_rounds} rounds."
                    ),
                },
            )
            return
        round_number = goal.rounds_started + 1
        content = render_goal_round_prompt(goal, round_number)
        message = create_user_message(
            content=content,
            source={
                "kind": "goal",
                "goalId": goal.id,
                "revision": goal.revision,
                "round": round_number,
            },
        )
        state.attempt = _Round(
            goal.id, goal.revision, round_number, message["id"], content
        )
        try:
            agent.followup(message)
        except Exception as error:  # noqa: BLE001 -- a failed queue blocks or disarms the goal
            state.attempt = None
            latest = self._current_goal(agent)
            if (
                latest is not None
                and latest.id == goal.id
                and latest.revision == goal.revision
                and self._drivable(latest)
            ):
                self.ctx.goals.block(
                    agent,
                    {"id": latest.id, "revision": latest.revision},
                    {
                        "code": QUEUE_FAILED_CODE,
                        "message": f"Could not queue goal round {round_number}: {error}",
                    },
                )
            else:
                self._disarm(agent)

    def request_drive(self, agent: Any) -> None:
        """Coalesce triggers onto one agent-local serialized driver."""
        state = self._state(agent)
        if state.stopping:
            return
        state.requested = True
        if state.run is not None:
            return

        async def run() -> None:
            while state.requested and not state.stopping:
                state.requested = False
                try:
                    await self.drive(agent, state)
                except Exception:  # noqa: BLE001 -- a driver failure disarms the goal
                    self._disarm(agent)

        try:
            state.run = asyncio.create_task(run())
        except RuntimeError:
            self._disarm(agent)
            return

        state.run.add_done_callback(lambda task: self._reschedule(state, agent))

    def _reschedule(self, state: _State, agent: Any) -> None:
        """Retire a completed driver task and start another if requested."""
        state.run = None
        if state.requested and not state.stopping:
            self.request_drive(agent)

    async def init(self) -> Callable[[], None]:
        """Install the event listeners that trigger and fence the driver."""
        disposers: list[Any] = []

        def on(name: str, handler: Any, *, waterfall: bool = False) -> None:
            if waterfall:
                disposers.append(self.ctx.on(name, handler, mode=EventMode.WATERFALL))
            else:
                disposers.append(self.ctx.on(name, handler))

        def on_status(payload: Mapping[str, Any]) -> None:
            agent = payload.get("agent")
            if agent is None:
                return
            state = self._state(agent)
            if payload.get("status") == "idle":
                state.competing_queued = False
                attempt = state.attempt
                if attempt is not None and attempt.phase == "queued":
                    inbox = getattr(agent, "inbox", None)
                    remove = getattr(inbox, "remove", None)
                    if callable(remove):
                        remove(attempt.message_id)
                    state.attempt = None
                self.request_drive(agent)

        def on_error(payload: Mapping[str, Any]) -> None:
            agent = payload.get("agent")
            if agent is not None:
                self._disarm(agent)

        def on_disposed(payload: Mapping[str, Any]) -> None:
            agent = payload.get("agent")
            if agent is not None:
                identifier = str(getattr(agent, "id", None) or id(agent))
                self._states.pop(identifier, None)

        def on_created(payload: Mapping[str, Any]) -> None:
            agent = payload.get("agent")
            if agent is not None:
                state = self._state(agent)
                state.attempt = None
                state.competing_queued = False
                state.needs_checkpoint = False

        def on_inbox(payload: Mapping[str, Any]) -> None:
            agent = payload.get("agent")
            message = payload.get("message")
            if agent is None or message is None:
                return
            state = self._state(agent)
            attempt = state.attempt
            if self._same_queued(
                message.get("content", []), message.get("source", {}), attempt
            ):
                if payload.get("type") == "claimed" and attempt is not None:
                    attempt.phase = "claimed"
                elif payload.get("type") == "discarded" and attempt is not None:
                    attempt.cancelled = True
                return
            state.competing_queued = True
            if attempt is not None and attempt.phase == "queued":
                attempt.stale = True
                state.attempt = None

        on("agent/status", on_status)
        on("agent/error", on_error)
        on("agent/disposed", on_disposed)
        on("agent/created", on_created)
        on(
            "agent/inbox/inserted",
            lambda payload: on_inbox({**payload, "type": "inserted"}),
        )
        on(
            "agent/inbox/claimed",
            lambda payload: on_inbox({**payload, "type": "claimed"}),
        )
        on(
            "agent/inbox/discarded",
            lambda payload: on_inbox({**payload, "type": "discarded"}),
        )

        def on_session_event(session: Any, event: Any) -> None:
            registry = self.ctx.get("agents", None, strict=False)
            agent = registry.get(session.session_id) if registry is not None else None
            if agent is None:
                return
            state = self._state(agent)
            if event.type == "goal/change":
                state.needs_checkpoint = True
                self.request_drive(agent)
            elif event.type == "user/message":
                attempt = state.attempt
                if (
                    attempt is not None
                    and event.payload.get("id") == attempt.message_id
                ):
                    attempt.phase = "admitted"
            elif event.type == "turn/end":
                reason = event.payload.get("reason", {})
                kind = reason.get("kind")
                if kind == "max-tokens":
                    self._disarm(agent)
                    return
                if kind != "aborted":
                    return
                abort_reason = reason.get("reason") or {}
                if abort_reason.get("kind") == "disposed":
                    self._disarm(agent)
                    return
                attempt = state.attempt
                if attempt is None or attempt.phase not in ("claimed", "admitted"):
                    self._disarm(agent)
                    return
                attempt.cancelled = True

        disposers.append(self.ctx.on("session/event", on_session_event))

        def on_pre_step(payload: Mapping[str, Any], next_fn: Any) -> Any:
            agent = payload.get("agent")
            messages = list(payload.get("messages") or [])
            submitted = next(
                (
                    message
                    for message in messages
                    if isinstance(message.get("source"), Mapping)
                    and message["source"].get("kind") == "goal"
                    and isinstance(message["source"].get("round"), int)
                    and message["source"]["round"] > 0
                ),
                None,
            )
            if submitted is None:
                return next_fn()
            state = self._state(agent)
            source = submitted["source"]
            goal = self._current_goal(agent)

            async def wrap() -> Any:
                valid = self._valid_reservation(
                    state, submitted.get("content", []), source, goal
                )
                if not valid:
                    attempt = state.attempt
                    if attempt is not None and self._same_round(source, attempt):
                        attempt.stale = True
                        state.attempt = None
                    self._restore_other_claimed(agent, messages, submitted.get("id"))
                    self.request_drive(agent)
                    return {"kind": "reject"}
                decision = next_fn()
                if hasattr(decision, "__await__"):
                    decision = await decision
                if decision.get("kind") == "reject":
                    state.attempt = None
                    assert goal is not None
                    self.ctx.goals.block(
                        agent,
                        {"id": goal.id, "revision": goal.revision},
                        {
                            "code": PROMPT_REJECTED_CODE,
                            "message": "Goal round was rejected before entering its step.",
                        },
                    )
                    return decision
                if not self._valid_reservation(
                    state, submitted.get("content", []), source, goal
                ):
                    state.attempt = None
                    self._restore_other_claimed(
                        agent, decision.get("messages", []), submitted.get("id")
                    )
                    self.request_drive(agent)
                    return {"kind": "reject"}
                return {**decision, "startsRequestSeries": True}

            return wrap()

        on("agent/pre-step", on_pre_step, waterfall=True)

        def dispose() -> None:
            for state in list(self._states.values()):
                state.stopping = True
                state.requested = False
                self._disarm(state.agent)
                if state.run is not None:
                    state.run.cancel()
            for disposer in disposers:
                disposer()

        return dispose

    @staticmethod
    def _same_round(source: Mapping[str, Any], attempt: _Round) -> bool:
        return (
            source.get("goalId") == attempt.goal_id
            and source.get("revision") == attempt.revision
            and source.get("round") == attempt.round
        )

    @staticmethod
    def _same_queued(content: Any, source: Any, attempt: _Round | None) -> bool:
        if attempt is None or not isinstance(source, Mapping):
            return False
        return (
            source.get("kind") == "goal"
            and GoalRoundDriver._same_round(source, attempt)
            and content == attempt.content
        )


def apply(ctx: Context, config: Mapping[str, Any] | None = None) -> GoalRoundDriver:
    """Mount ``ctx.goalRoundDriver``."""
    return GoalRoundDriver(ctx, config)


apply.inject = ["agents", "goals", "sessions"]  # type: ignore[attr-defined]

__all__ = [
    "PROMPT_REJECTED_CODE",
    "QUEUE_FAILED_CODE",
    "ROUND_LIMIT_CODE",
    "GoalRoundDriver",
    "apply",
    "render_goal_round_prompt",
]
