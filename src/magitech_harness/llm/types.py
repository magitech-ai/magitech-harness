"""LLM models, canonical message/stream vocabularies, and legacy types."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.session.events import ToolCallData


class UsageData(BaseModel):
    """Token consumption telemetry (legacy Phase 2 representation)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0


class ModelResponse(BaseModel):
    """Non-streaming model response payload (legacy Phase 2 representation)."""

    text: str = ""
    tool_calls: list[ToolCallData] = Field(default_factory=list)
    finish_reason: str = "stop"
    raw_response: dict[str, Any] = Field(default_factory=dict)
    usage: UsageData = Field(default_factory=UsageData)


class ModelChunk(BaseModel):
    """Incremental model response chunk (legacy Phase 2 representation)."""

    text: str = ""
    tool_calls: list[ToolCallData] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: UsageData = Field(default_factory=UsageData)


ImageMediaType = Literal["image/png", "image/jpeg", "image/webp", "image/gif"]
ModelModality = Literal["text", "image"]
SystemPromptUpdate = Literal["in-history"]
ToolUpdate = Literal["in-history", "addition-only"]
LlmModelBadgeTone = Literal["neutral", "info", "success", "warning"]


class ImageAttachmentRef(TypedDict):
    """Durable normalized image reference stored by the attachment service."""

    attachmentId: str
    mediaType: ImageMediaType
    bytes: int
    width: int
    height: int
    name: NotRequired[str]


class ImageRequestTarget(TypedDict):
    """Dimensions and maximum byte limit applied to request images."""

    width: int
    height: int
    maxBytes: int


class FileAttachmentRef(TypedDict):
    """Durable verbatim file reference stored by the attachment service."""

    attachmentId: str
    name: str
    bytes: int
    mediaType: NotRequired[str]


class RequestImageAttachment(TypedDict):
    """Resolved request image preview dimensions."""

    width: int
    height: int


class LlmFailure(TypedDict):
    """Serializable provider or transport failure facts."""

    message: str
    code: str
    status: NotRequired[int]
    providerRetryAfterMs: NotRequired[float]
    requestId: NotRequired[str]
    offloadImages: NotRequired[int]


class TextBlock(TypedDict):
    """Plain text content block."""

    type: Literal["text"]
    text: str


class ReasoningBlock(TypedDict):
    """Model reasoning / thinking content block."""

    type: Literal["reasoning"]
    text: str


class ImageBlock(TypedDict):
    """Durable image reference block with optional offload marker."""

    type: Literal["image"]
    attachment: ImageAttachmentRef
    offloaded: NotRequired[Literal[True]]


class FileBlock(TypedDict):
    """Durable file reference block."""

    type: Literal["file"]
    attachment: FileAttachmentRef


class ToolCallBlock(TypedDict):
    """Tool invocation content block emitted by the model."""

    type: Literal["tool-call"]
    id: str
    name: str
    arguments: str


class ToolAdditionBlock(TypedDict):
    """Developer message block activating a deferred tool schema."""

    type: Literal["tool-addition"]
    toolName: str


class ToolRemovalBlock(TypedDict):
    """Developer message block deactivating a session tool."""

    type: Literal["tool-removal"]
    toolName: str


ContentBlock = (
    TextBlock
    | ReasoningBlock
    | ImageBlock
    | FileBlock
    | ToolCallBlock
    | ToolAdditionBlock
    | ToolRemovalBlock
    | dict[str, Any]
)


class TokenUsage(TypedDict):
    """Disjoint token accounting for one model call."""

    inputTokens: int
    outputTokens: int
    totalTokens: NotRequired[int]
    cacheReadTokens: NotRequired[int]
    cacheWriteTokens: NotRequired[int]
    reasoningTokens: NotRequired[int]


class ReplayEnvelope(TypedDict):
    """Adapter-private replay metadata aligned with emitted blocks."""

    response: Any
    blocks: NotRequired[Sequence[Any]]


FinishReason = dict[str, Any]
StreamChunk = dict[str, Any]


class LlmImageRequestPrice(TypedDict):
    """Request visual tokens and text emitted for one image occurrence."""

    visualTokens: int
    text: str


class LlmImageRequestPricing(TypedDict):
    """Synchronous provider pricing hook for image occurrences."""

    priceImages: Callable[[Sequence[ImageBlock]], Sequence[LlmImageRequestPrice]]


class LlmImageRequestBudget(TypedDict):
    """Request-image byte and count budget enforced per route."""

    representation: Literal["raw", "base64"]
    maxBytes: NotRequired[int]
    maxImages: NotRequired[int]
    byteQuantum: NotRequired[int]
    countQuantum: NotRequired[int]


class LlmProviderInfo(TypedDict):
    """Display metadata for one registered provider route."""

    id: str
    name: str


class LlmNativeAuth(TypedDict):
    """Native authentication fallback metadata for a configurable provider."""

    env: Sequence[str]
    oauth: bool


class LlmConfigurableProvider(TypedDict):
    """Directory declaration for one provider route configurable in settings."""

    provider: str
    displayName: str
    settingsNs: str
    settingsPath: Sequence[str]
    declared: NotRequired[bool]
    defaultBaseURL: NotRequired[str]
    baseURLEnv: NotRequired[Sequence[str]]
    nativeAuth: NotRequired[LlmNativeAuth]
    error: NotRequired[str]


class LlmModelDiscoveryRequest(TypedDict, total=False):
    """Draft or existing provider endpoint interrogation parameters."""

    provider: str
    baseURL: str
    api: str
    apiKey: str


class LlmDiscoveredModel(TypedDict):
    """Model metadata advertised by a provider endpoint."""

    id: str
    name: NotRequired[str]
    contextWindow: NotRequired[int]
    maxTokens: NotRequired[int]
    inputModalities: NotRequired[Sequence[ModelModality]]


class LlmModelBadge(TypedDict):
    """Selector badge attached to an advertised model."""

    text: str | dict[str, str]
    tone: LlmModelBadgeTone


class LlmModelContext(TypedDict):
    """Provider context window metadata."""

    contextWindow: int


class LlmReasoningEffortInfo(TypedDict):
    """Display metadata for one selectable reasoning effort level."""

    id: str
    name: str
    description: NotRequired[str]


class LlmModelReasoningInfo(TypedDict):
    """Selectable reasoning effort catalog and optional default."""

    efforts: Sequence[LlmReasoningEffortInfo]
    defaultEffort: NotRequired[str]


class LlmModelInfo(TypedDict):
    """Adapter-discovered catalog model entry."""

    provider: str
    id: str
    name: str
    description: NotRequired[str]
    inputModalities: NotRequired[Sequence[ModelModality]]
    badge: NotRequired[LlmModelBadge]


class LlmResolvedModelInfo(TypedDict):
    """Exact-route model metadata resolved from an adapter."""

    provider: str
    id: str
    name: str
    description: NotRequired[str]
    inputModalities: NotRequired[Sequence[ModelModality]]
    context: NotRequired[LlmModelContext]
    defaultMaxTokens: NotRequired[int]
    reasoning: NotRequired[LlmModelReasoningInfo]
    systemPromptUpdate: NotRequired[SystemPromptUpdate]
    toolUpdate: NotRequired[ToolUpdate]


class ToolSchema(TypedDict):
    """JSON-schema description of one tool exposed to a model."""

    deferLoading: NotRequired[Literal[True]]
    name: str
    description: str
    parameters: dict[str, Any]


class ToolHistoryUpdate(TypedDict):
    """One recorded developer message and its newly added tool schemas."""

    messageId: str
    additions: Sequence[ToolSchema]


class ToolHistory(TypedDict):
    """Folded tool declarations and chronological update entries."""

    tools: Sequence[ToolSchema]
    updates: Sequence[ToolHistoryUpdate]


RequestMessage = Mapping[str, Any]


class GenerateOptions(TypedDict, total=False):
    """Complete model request envelope dispatched through LLMService.stream."""

    provider: str
    model: str
    reasoningEffort: str
    messages: Sequence[RequestMessage]
    system: str
    tools: Sequence[ToolSchema]
    toolHistory: ToolHistory
    temperature: float
    maxTokens: int
    stop: Sequence[str]
    signal: AbortSignal
    sessionId: str
    purpose: str
