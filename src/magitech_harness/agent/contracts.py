"""Public agent contracts, runtime option types, and lifecycle event vocabulary."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal, NotRequired, Protocol, TypedDict, runtime_checkable

from animas import Context

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.session.store import Session

InboxTarget = Literal["next-turn", "next-step"]
SessionStartSource = Literal["startup", "resume", "clear", "compact"]
AgentStatus = Literal["idle", "running"]

AGENT_EVENT_TYPES: tuple[str, ...] = (
    "agent/created",
    "agent/disposed",
    "agent/status",
    "agent/inbox/inserted",
    "agent/inbox/claimed",
    "agent/inbox/discarded",
    "agent/pre-step",
    "agent/request",
    "agent/request-error",
    "agent/assistant-stream",
    "agent/turn-stopping",
    "agent/error",
)


class StepBoundary(TypedDict):
    """Latest durable step boundary marker."""

    kind: Literal["start", "end"]
    seq: int


class TurnBoundaryProjection(TypedDict):
    """Turn and step positions recorded in a Session."""

    openTurnStartSeq: int | None
    lastStepStartSeq: int | None
    lastStepBoundary: StepBoundary | None
    lastTurn: int


class ModelSelection(TypedDict):
    """Provider, model, and optional reasoning effort chosen for a Session."""

    provider: str
    model: str
    reasoningEffort: NotRequired[str]


class ModelSelectionProjection(TypedDict):
    """Client view of the durable model-selection fold."""

    lastUsed: ModelSelection | None
    next: ModelSelection | None


class AgentOptions(TypedDict, total=False):
    """Route fallback and output controls for a live Agent."""

    provider: str
    model: str
    reasoningEffort: str | None
    maxTokens: int


class CancelOptions(TypedDict, total=False):
    """Options controlling whether Agent.cancel keeps queued inbox input."""

    keepInbox: bool | None


InboxState = dict[str, list[dict[str, Any]]]
PreStepDecision = dict[str, Any]
RequestErrorAction = dict[str, Any] | None
AssistantStreamFrame = dict[str, Any]


class AgentMeta(TypedDict, total=False):
    """Durable session creation metadata accepted by create_agent."""

    cwd: str
    parentSession: str
    isSeeded: bool
    origin: Literal["subagent"]
    delegationDepth: int
    agentPreset: str


@runtime_checkable
class AgentSetupCommit(Protocol):
    """Synchronous publication finalizer returned by unpublished Agent setup."""

    def commit(self) -> None:
        """Validate and commit prepared setup right before publication."""
        ...


AgentSetup = Callable[
    [Context, "Agent"],
    AgentSetupCommit | None | Awaitable[AgentSetupCommit | None],
]


class CreateAgentOptions(TypedDict, total=False):
    """Options for programmatically creating an agent through AgentRegistry."""

    sessionId: str
    meta: AgentMeta
    inheritedEventCount: int
    seed: Sequence[Any]
    agentOptions: AgentOptions
    signal: AbortSignal
    setup: AgentSetup


class ResumeAgentOptions(TypedDict, total=False):
    """Options for resuming an agent on a persisted session."""

    resumeSessionId: str
    agentOptions: AgentOptions
    signal: AbortSignal
    setup: AgentSetup


@runtime_checkable
class Model(Protocol):
    """Agent-owned access to durable model selection for future steps."""

    def current(self) -> ModelSelection | None:
        """Resolve the choice for the next step without mutating state."""
        ...

    def select(self, selection: ModelSelection) -> None:
        """Record a complete model selection for future steps."""
        ...


@runtime_checkable
class Inbox(Protocol):
    """Agent-owned access to durable pending work."""

    @property
    def next_turn(self) -> list[dict[str, Any]]:
        """Prompts awaiting individual turns."""
        ...

    @property
    def next_step(self) -> list[dict[str, Any]]:
        """Input awaiting the next step boundary."""
        ...

    def clear(self) -> None:
        """Durably cancel all pending input, clearing next-step before next-turn."""
        ...

    def append(self, target: InboxTarget, message: dict[str, Any]) -> None:
        """Append one message to a pending list."""
        ...

    def prepend(self, target: InboxTarget, message: dict[str, Any]) -> None:
        """Prepend one message to a pending list."""
        ...

    def replace(self, message_id: str, new_message: dict[str, Any]) -> bool:
        """Replace one pending message in place by id."""
        ...

    def remove(self, message_id: str) -> bool:
        """Remove one pending message by id."""
        ...

    def splice(
        self,
        target: InboxTarget,
        start: float,
        delete_count: float,
        inserted: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply splice semantics and durably record the normalized result."""
        ...


@runtime_checkable
class Agent(Protocol):
    """Live session-backed agent handle."""

    @property
    def id(self) -> str: ...

    @property
    def options(self) -> AgentOptions: ...

    @property
    def session(self) -> Session: ...

    @property
    def model(self) -> Model: ...

    @property
    def inbox(self) -> Inbox: ...

    @property
    def status(self) -> AgentStatus: ...

    @property
    def ctx(self) -> Context: ...

    def cancel(
        self, cause: dict[str, Any], options: CancelOptions | None = None
    ) -> None: ...

    def when_idle(self) -> Awaitable[None]: ...

    def run_maintenance(
        self, task: Callable[[AbortSignal], Awaitable[Any]]
    ) -> Awaitable[Any]: ...

    def send(
        self, message: dict[str, Any], target: InboxTarget, wakeup: bool
    ) -> None: ...

    def followup(self, message: dict[str, Any]) -> None: ...

    def steer(self, message: dict[str, Any]) -> None: ...

    def inject(self, message: dict[str, Any]) -> None: ...


class AgentHandle(TypedDict):
    """Owned live agent handle and its async disposer."""

    agent: Agent
    dispose: Callable[[], Awaitable[None] | None]


@runtime_checkable
class AgentFactory(Protocol):
    """Factory registered with AgentRegistry to construct and resume agents."""

    def create_agent(
        self, owner_ctx: Context, options: CreateAgentOptions
    ) -> Awaitable[AgentHandle]:
        """Create and publish a new agent attached to owner_ctx."""
        ...

    def resume(
        self, owner_ctx: Context, options: ResumeAgentOptions
    ) -> Awaitable[AgentHandle]:
        """Load a persisted session and resume an agent attached to owner_ctx."""
        ...
