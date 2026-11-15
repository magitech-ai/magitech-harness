"""Tool service contracts, errors, pipeline interfaces, and batch types."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal, NotRequired, Protocol, TypedDict, runtime_checkable

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.llm.error import HarnessError
from magitech_harness.llm.types import ContentBlock, ToolCallBlock, ToolSchema
from magitech_harness.tools.presentation import ToolCallView, ToolResultView

RUN_CODE_NAME = "run_code"
TOOL_ABORTED = "ABORTED"
TOOL_ABORTED_BEFORE_DISPATCH = "ABORTED_BEFORE_DISPATCH"

ToolExecutionToken = object


class ToolActivationError(HarnessError):
    """Raised when a conditional tool's activation predicate fails."""

    def __init__(self, name: str, cause: Any = None) -> None:
        super().__init__(
            f'cannot resolve availability of tool "{name}"',
            "TOOL_ACTIVATION_FAILED",
            cause=cause,
        )
        self.tool_name = name


class ToolNotFoundError(HarnessError):
    """Raised when resolving or running an unknown or unpermitted tool."""

    def __init__(self, tool_name: str, reachable_from: str | None = None) -> None:
        message = (
            f'unknown tool "{tool_name}"'
            if reachable_from is None
            else f'unknown tool "{tool_name}": {reachable_from}'
        )
        super().__init__(message, "UNKNOWN_TOOL")
        self.tool_name = tool_name


class ToolOutputError(HarnessError):
    """Raised when a tool output violates its declared output schema or losslessness."""

    def __init__(self, tool_name: str, violations: Sequence[str] | str) -> None:
        if isinstance(violations, str):
            self.violations: list[str] = [violations]
            detail = violations
        else:
            self.violations = list(violations)
            detail = "; ".join(self.violations)
        super().__init__(
            f'tool "{tool_name}" returned invalid output: {detail}',
            "INVALID_TOOL_OUTPUT",
        )
        self.tool_name = tool_name


class ToolExecutorUnavailableError(HarnessError):
    """Raised when execute() is invoked with no active scoped/global executor."""

    def __init__(
        self,
        message: str = (
            "The selected tool executor is unavailable. "
            "Load an executor in this composition before running the agent."
        ),
    ) -> None:
        super().__init__(message, "TOOL_EXECUTOR_UNAVAILABLE")


class ToolBatchExecutionError(Exception):
    """Raised when a batch scheduler fails mid-batch, preserving committed outcomes."""

    def __init__(
        self,
        cause: Any,
        partial: ToolBatchResult,
        message: str = "Tool batch execution failed",
    ) -> None:
        super().__init__(message)
        self.partial: ToolBatchResult = partial
        self.cause = cause
        if isinstance(cause, BaseException):
            self.__cause__ = cause


class ToolResult(TypedDict):
    """Completed outcome handed to ToolDefinition.presentResult."""

    content: list[ContentBlock]
    isError: bool
    meta: NotRequired[Any]


class ToolOutputDefinition(TypedDict):
    """Declared schema, content renderer, and optional presentation metadata projector."""

    schema: dict[str, Any]
    render: Callable[[Any, Any], Sequence[ContentBlock]]
    presentationMeta: NotRequired[Callable[[Any, Any], Any]]


class ToolActivationContext(TypedDict):
    """Snapshot context passed to a conditional tool's ``when`` predicate."""

    agent: Any
    modelSelection: Any
    signal: AbortSignal


class ToolRegistrationOptions(TypedDict, total=False):
    """Optional conditional activation descriptor for ``register()``."""

    when: Callable[[ToolActivationContext], bool | Awaitable[bool]]


class ToolDefinition(TypedDict):
    """Registered tool contract carrying validated schemas, output, and callbacks."""

    name: str
    description: str
    parameters: dict[str, Any]
    output: ToolOutputDefinition
    execute: Callable[[Any, ToolRunContext], Any]
    projectContent: NotRequired[
        Callable[[ToolExecution, ToolExecutionResult], Sequence[ContentBlock] | None]
    ]
    finalizeContent: NotRequired[
        Callable[[ToolExecution, ToolExecutionResult], Sequence[ContentBlock] | None]
    ]
    timeoutMs: NotRequired[float]
    isConcurrencySafe: NotRequired[Callable[[Any], bool]]
    presentCall: NotRequired[Callable[[Any], ToolCallView | None]]
    presentResult: NotRequired[Callable[[Any, ToolResult], ToolResultView | None]]
    deferLoading: NotRequired[bool]


class ToolExecutionMode(TypedDict):
    """Per-call concurrency classification."""

    kind: Literal["parallel", "exclusive"]


class ToolExecutionInput(TypedDict):
    """Single-call input accepted by ToolExecutionPipeline."""

    callId: str
    rootCallId: NotRequired[str]
    name: str
    schema: NotRequired[ToolSchema]
    arguments: Any
    agent: NotRequired[Any]
    toolSet: NotRequired[ResolvedToolSet]
    parent: NotRequired[Any]
    signal: AbortSignal


class ToolExecution(TypedDict):
    """Readonly-to-observers execution descriptor carrying pipeline token."""

    callId: str
    rootCallId: str
    name: str
    schema: NotRequired[ToolSchema]
    arguments: Any
    agent: NotRequired[Any]
    toolSet: NotRequired[ResolvedToolSet]
    parent: NotRequired[Any]
    signal: AbortSignal
    token: Any


ToolDispatchExecution = ToolExecution


class ToolRunContext(ToolExecution):
    """Execution context handed to a tool body."""

    deferContext: Callable[[Mapping[str, Any]], None]
    concludeTurn: Callable[[], None]


class ToolErrorInfo(TypedDict, total=False):
    """Machine-routable failure class attached to a failed tool result."""

    name: str
    code: str
    reason: str


class ToolFailure(TypedDict):
    """Failure descriptor carried on a failed ToolExecutionResult."""

    message: str
    info: NotRequired[ToolErrorInfo | dict[str, Any]]


class ToolExecutionSuccess(TypedDict):
    """Successful normalized tool execution result."""

    isError: Literal[False]
    value: Any
    content: list[ContentBlock]
    meta: NotRequired[Any]
    additionalContexts: NotRequired[list[Mapping[str, Any]]]
    concludesTurn: NotRequired[Literal[True]]


class ToolExecutionFailure(TypedDict):
    """Failed normalized tool execution result."""

    isError: Literal[True]
    error: ToolFailure
    content: list[ContentBlock]
    meta: NotRequired[Any]
    additionalContexts: NotRequired[list[Mapping[str, Any]]]


ToolExecutionResult = ToolExecutionSuccess | ToolExecutionFailure


PreToolDecision = (
    dict[str, Any]  # allow | deny | cancel | ask
)
PostToolDecision = (
    dict[str, Any]  # accept | block
)


class ToolRestriction(TypedDict, total=False):
    """Scoped allow/deny filter over inherited tools."""

    allow: Sequence[str]
    deny: Sequence[str]


ToolGuard = Callable[[ToolExecution], str | None]


class ToolCallResult(TypedDict):
    """Batch entry mapping one call id to its finished result."""

    callId: str
    result: ToolExecutionResult


class ToolBatchLog(TypedDict):
    """Committed model step identifiers for tool/call and tool/result logging."""

    turn: int
    step: int


class ToolBatchInput(TypedDict):
    """Batch invocation submitted to ToolsService.execute / ToolExecutor.execute."""

    calls: Sequence[ToolCallBlock]
    signal: AbortSignal
    maxParallelToolCalls: NotRequired[int]
    agent: NotRequired[Any]
    toolSet: NotRequired[ResolvedToolSet]
    log: NotRequired[ToolBatchLog]


class ToolBatchResult(TypedDict):
    """Settled batch response carrying input-ordered results and turn signals."""

    results: list[ToolCallResult]
    additionalContexts: list[Mapping[str, Any]]
    concluded: bool


class ToolModelInput(TypedDict):
    """Schema projection and optional known-names set returned by an executor."""

    schemas: list[ToolSchema]
    knownNames: NotRequired[list[str]]


class ToolPromptSection(TypedDict):
    """Optional executor-owned prompt section attached during set_executor."""

    name: str
    order: int | float
    text: str | Callable[[Sequence[ToolDefinition]], str]
    interpolate: NotRequired[Any]


@runtime_checkable
class ToolExecutor(Protocol):
    """Pluggable executor responsible for schema projection and batch dispatch."""

    def describe(self, definitions: Sequence[ToolDefinition]) -> ToolModelInput:
        """Project tool definitions into model-visible schemas."""
        ...

    def execute(self, batch: ToolBatchInput) -> Awaitable[ToolBatchResult]:
        """Execute a batch of model tool calls."""
        ...


@runtime_checkable
class ResolvedToolSet(Protocol):
    """Per-step admitted tool snapshot bound to one agent."""

    def dispose(self) -> None:
        """Permanently close admission for subsequent tool lookups."""
        ...

    def view(self, scope: Any) -> Mapping[str, ToolDefinition]:
        """Return admitted registrations visible in their original scope."""
        ...


__all__ = [
    "RUN_CODE_NAME",
    "TOOL_ABORTED",
    "TOOL_ABORTED_BEFORE_DISPATCH",
    "PostToolDecision",
    "PreToolDecision",
    "ResolvedToolSet",
    "ToolActivationContext",
    "ToolActivationError",
    "ToolBatchExecutionError",
    "ToolBatchInput",
    "ToolBatchLog",
    "ToolBatchResult",
    "ToolCallResult",
    "ToolDefinition",
    "ToolDispatchExecution",
    "ToolErrorInfo",
    "ToolExecution",
    "ToolExecutionFailure",
    "ToolExecutionInput",
    "ToolExecutionMode",
    "ToolExecutionResult",
    "ToolExecutionSuccess",
    "ToolExecutionToken",
    "ToolExecutor",
    "ToolExecutorUnavailableError",
    "ToolFailure",
    "ToolGuard",
    "ToolModelInput",
    "ToolNotFoundError",
    "ToolOutputDefinition",
    "ToolOutputError",
    "ToolPromptSection",
    "ToolRegistrationOptions",
    "ToolRestriction",
    "ToolResult",
    "ToolRunContext",
]
