"""Agent subsystem exports: contracts, projections, inbox, model selection, registry, and loop."""

from magitech_harness.agent.agent_loop import (
    CONFIGURED_AGENT_IDENTITIES_KEY,
    AgentLoop,
    FactoryOwnership,
    apply_launcher_identities,
    race_abort,
    race_abort_call,
    validate_configured_agents,
)
from magitech_harness.agent.archive_admission import install_turn_archive_admission
from magitech_harness.agent.assistant_stream import AssistantStreamAttempt
from magitech_harness.agent.cancellation import AbortController, AbortError, AbortSignal
from magitech_harness.agent.constants import DEFAULT_MAX_PARALLEL_TOOL_CALLS
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
from magitech_harness.agent.react_loop import ReactLoopAgent
from magitech_harness.agent.registry import AgentRegistry
from magitech_harness.agent.runtime_context import RuntimeContextProjection

__all__ = [
    "AGENT_EVENT_TYPES",
    "CONFIGURED_AGENT_IDENTITIES_KEY",
    "DEFAULT_MAX_PARALLEL_TOOL_CALLS",
    "AbortController",
    "AbortError",
    "AbortSignal",
    "Agent",
    "AgentEventDispatch",
    "AgentFactory",
    "AgentHandle",
    "AgentInbox",
    "AgentLoop",
    "AgentLoopService",
    "AgentMeta",
    "AgentModel",
    "AgentOptions",
    "AgentRegistry",
    "AgentSetup",
    "AgentSetupCommit",
    "AgentStatus",
    "AssistantStreamAttempt",
    "AssistantStreamFrame",
    "CancelOptions",
    "ConsumedWork",
    "CreateAgentOptions",
    "FactoryOwnership",
    "Inbox",
    "InboxState",
    "InboxTarget",
    "Model",
    "ModelSelection",
    "ModelSelectionProjection",
    "PreStepDecision",
    "ReactLoopAgent",
    "ReactLoopInbox",
    "RequestErrorAction",
    "ResumeAgentOptions",
    "RuntimeContextProjection",
    "SessionStartSource",
    "StepBoundary",
    "TurnBoundaryProjection",
    "agent_carrier",
    "agent_events",
    "apply_launcher_identities",
    "assemble_context_for",
    "emit_agent_event",
    "fold_consumed_work",
    "inbox_projection_definition",
    "install_model_selection",
    "install_turn_archive_admission",
    "model_selection_projection_definition",
    "race_abort",
    "race_abort_call",
    "turn_boundary_projection_definition",
    "validate_configured_agents",
]
