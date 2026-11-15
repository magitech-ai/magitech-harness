"""Agent subsystem exports: contracts, projections, inbox, model selection, registry, and loop."""

from magitech_harness.agent.archive_admission import install_turn_archive_admission
from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.agent.consumed_work import ConsumedWork, fold_consumed_work
from magitech_harness.agent.contracts import (
    AGENT_EVENT_TYPES,
    Agent,
    AgentFactory,
    AgentHandle,
    AgentMeta,
    AgentOptions,
    AgentSetup,
    AgentSetupCommit,
    AgentStatus,
    AssistantStreamFrame,
    CancelOptions,
    CreateAgentOptions,
    Inbox,
    InboxState,
    InboxTarget,
    Model,
    ModelSelection,
    ModelSelectionProjection,
    PreStepDecision,
    RequestErrorAction,
    ResumeAgentOptions,
    SessionStartSource,
    StepBoundary,
    TurnBoundaryProjection,
)
from magitech_harness.agent.dispatch import (
    AgentEventDispatch,
    agent_carrier,
    agent_events,
    assemble_context_for,
    emit_agent_event,
)
from magitech_harness.agent.inbox import (
    AgentInbox,
    ReactLoopInbox,
    inbox_projection_definition,
)
from magitech_harness.agent.loop import AgentLoopService
from magitech_harness.agent.model_selection import (
    AgentModel,
    install_model_selection,
    model_selection_projection_definition,
)
from magitech_harness.agent.projection import turn_boundary_projection_definition
from magitech_harness.agent.registry import AgentRegistry

__all__ = [
    "AGENT_EVENT_TYPES",
    "AbortError",
    "AbortSignal",
    "Agent",
    "AgentEventDispatch",
    "AgentFactory",
    "AgentHandle",
    "AgentInbox",
    "AgentLoopService",
    "AgentMeta",
    "AgentModel",
    "AgentOptions",
    "AgentRegistry",
    "AgentSetup",
    "AgentSetupCommit",
    "AgentStatus",
    "AssistantStreamFrame",
    "CancelOptions",
    "ConsumedWork",
    "CreateAgentOptions",
    "Inbox",
    "InboxState",
    "InboxTarget",
    "Model",
    "ModelSelection",
    "ModelSelectionProjection",
    "PreStepDecision",
    "ReactLoopInbox",
    "RequestErrorAction",
    "ResumeAgentOptions",
    "SessionStartSource",
    "StepBoundary",
    "TurnBoundaryProjection",
    "agent_carrier",
    "agent_events",
    "assemble_context_for",
    "emit_agent_event",
    "fold_consumed_work",
    "inbox_projection_definition",
    "install_model_selection",
    "install_turn_archive_admission",
    "model_selection_projection_definition",
    "turn_boundary_projection_definition",
]
