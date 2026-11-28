"""Newline-delimited JSON-RPC 2.0 and the named SDK method vocabulary.

Port of `packages/sdk/protocol` at the pinned dsh reference. Frames are one
JSON object per newline; a frame with both ``id`` and ``method`` is a request,
``id`` alone is a response, and ``method`` alone is a notification. Malformed
lines are ignored, and error responses preserve the peer's wire ``code`` and
optional ``data``. Transport lifecycle never destroys caller-owned streams.
"""

import asyncio
import contextlib
import json
import uuid
from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import Any, Literal, NotRequired, Required, TypedDict, cast

from magitech_harness.agent.cancellation import AbortError, AbortSignal

JsonRpcId = str | int | float
METHOD_NOT_FOUND = -32601
INTERNAL_ERROR = -32603


class JsonRpcResponseError(Exception):
    """An error response that preserves the wire code and optional data."""

    # Wire codes are JSON numbers: integer and fractional values are both valid.
    def __init__(
        self,
        code: int | float | None,  # noqa: PYI041
        message: str,
        data: Any = None,
    ) -> None:
        self.code = code
        self.data = data
        super().__init__(message)


class JsonRpcFrame(TypedDict):
    jsonrpc: str
    method: str
    params: dict[str, Any]


class JsonRpcRequest(TypedDict):
    jsonrpc: str
    id: JsonRpcId
    method: str
    params: dict[str, Any]


class JsonRpcResponse(TypedDict):
    jsonrpc: str
    id: JsonRpcId
    result: Any


class JsonRpcErrorFrame(TypedDict):
    jsonrpc: str
    id: JsonRpcId
    error: dict[str, Any]


class JsonRpcNotification(TypedDict, total=False):
    jsonrpc: str
    method: str
    params: dict[str, Any]


class InitializeParams(TypedDict):
    cwd: Required[str]
    provider: Required[str]
    model: Required[str]
    approvalRequests: NotRequired[bool]
    reasoningEffort: NotRequired[str]
    maxTokens: NotRequired[int]
    capabilities: NotRequired[dict[str, Any]]


class InitializeResult(TypedDict):
    serverInfo: dict[str, str]


class PromptContentBlock(TypedDict, total=False):
    type: str
    text: str
    data: str
    mimeType: str
    name: str


class SdkEncodedImageBlock(TypedDict):
    type: str
    data: str
    mimeType: str
    name: NotRequired[str]


SdkPromptContentBlock = PromptContentBlock
SdkRunStatus = Literal["ok", "error"]


class SessionPromptParams(TypedDict):
    sessionId: str
    contentBlocks: list[PromptContentBlock]
    timeZone: NotRequired[str]


class SessionPromptResult(TypedDict):
    messageId: str


class SessionQueryParams(TypedDict):
    sessionId: str
    contentBlocks: list[PromptContentBlock]
    timeZone: NotRequired[str]


class SessionQueryResult(TypedDict, total=False):
    kind: str
    messageId: str
    execution: dict[str, Any]


class SessionResumeParams(TypedDict):
    sessionId: str
    permissionMode: NotRequired[str]


class SessionWaitParams(TypedDict):
    sessionId: str


class SessionWorkingDirectoryParams(TypedDict):
    sessionId: str


class SessionWorkingDirectorySetParams(TypedDict):
    sessionId: str
    path: str


class SessionWorkingDirectoryResult(TypedDict):
    cwd: str


class ApprovalRequestParams(TypedDict):
    sessionId: str
    toolName: str
    callId: NotRequired[str]
    reason: NotRequired[str]


class ApprovalRequestResult(TypedDict):
    outcome: Literal["allowed-once", "rejected", "cancelled"]


class SdkUserQuestionRequest(TypedDict):
    requestId: str
    sessionId: str
    questions: list[dict[str, Any]]


class SdkUserQuestionAnswerItem(TypedDict):
    id: str
    selected: list[str]
    custom: NotRequired[str]


class SdkUserQuestionAnswer(TypedDict):
    answers: list[SdkUserQuestionAnswerItem]


class SdkUserQuestionResult(TypedDict, total=False):
    answers: list[SdkUserQuestionAnswerItem]
    cancelled: bool


class SdkUserQuestionCancelled(TypedDict):
    requestId: str
    sessionId: str


class SessionEventNotification(TypedDict):
    sessionId: str
    event: dict[str, Any]


class SessionStatusNotification(TypedDict):
    sessionId: str
    status: Literal["idle", "running"]


class SessionQueryCompletedNotification(TypedDict):
    sessionId: str
    commandId: str
    finalResponse: str
    finishReason: str


class SubagentStartedNotification(TypedDict):
    parentSessionId: str
    childSessionId: str


class SubagentFinishedNotification(TypedDict):
    provider: str
    agentId: str
    parentSessionId: str
    childSessionId: str
    status: Literal["ok", "error"]
    stopReason: str
    lastAssistantMessage: NotRequired[list[PromptContentBlock]]


CLIENT_REQUEST_METHODS: tuple[str, ...] = (
    "initialize",
    "session/prompt",
    "session/query",
    "session/resume",
    "session/wait",
    "session/working-directory/get",
    "session/working-directory/set",
    "shutdown",
)
HOST_REQUEST_METHODS: tuple[str, ...] = ("approval/request", "user-question/ask")
NOTIFICATION_METHODS: tuple[str, ...] = (
    "user-question.cancelled",
    "session.event",
    "session.status",
    "session.query.completed",
    "subagent.started",
    "subagent.finished",
)

HarnessSdkRequestMap: Mapping[str, Any] = MappingProxyType(
    {
        "initialize": (InitializeParams, InitializeResult),
        "session/prompt": (SessionPromptParams, SessionPromptResult),
        "session/query": (SessionQueryParams, SessionQueryResult),
        "session/resume": (SessionResumeParams, dict),
        "session/wait": (SessionWaitParams, dict),
        "session/working-directory/get": (
            SessionWorkingDirectoryParams,
            SessionWorkingDirectoryResult,
        ),
        "session/working-directory/set": (
            SessionWorkingDirectorySetParams,
            SessionWorkingDirectoryResult,
        ),
        "shutdown": (dict, dict),
        "approval/request": (ApprovalRequestParams, ApprovalRequestResult),
        "user-question/ask": (SdkUserQuestionRequest, SdkUserQuestionResult),
    }
)
HarnessSdkClientRequestMap: Mapping[str, Any] = MappingProxyType(
    {method: HarnessSdkRequestMap[method] for method in CLIENT_REQUEST_METHODS}
)
HarnessSdkHostRequestMap: Mapping[str, Any] = MappingProxyType(
    {method: HarnessSdkRequestMap[method] for method in HOST_REQUEST_METHODS}
)
HarnessSdkNotificationMap: Mapping[str, Any] = MappingProxyType(
    {
        "user-question.cancelled": SdkUserQuestionCancelled,
        "session.event": SessionEventNotification,
        "session.status": SessionStatusNotification,
        "session.query.completed": SessionQueryCompletedNotification,
        "subagent.started": SubagentStartedNotification,
        "subagent.finished": SubagentFinishedNotification,
    }
)

RequestHandler = Callable[[str, dict[str, Any]], Awaitable[Any]]
NotificationHandler = Callable[[str, dict[str, Any]], None]


def _object_params(params: Any) -> dict[str, Any]:
    """Arrays and scalars collapse to an empty params object."""
    return params if isinstance(params, dict) else {}


def _valid_id(value: Any) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def _abort_error(reason: Any) -> AbortError:
    if isinstance(reason, BaseException):
        return AbortError(str(reason))
    return AbortError(f"JSON-RPC request aborted: {reason}")


class JsonRpcLineTransport:
    """Line-delimited JSON-RPC endpoint over caller-owned async byte streams."""

    def __init__(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._request_handler: RequestHandler | None = None
        self._notification_handler: NotificationHandler | None = None
        self._pending: dict[JsonRpcId, asyncio.Future[Any]] = {}
        self._failure: Exception | None = None
        self._task: asyncio.Task[None] | None = None
        self._started = False

    def on_request(self, handler: RequestHandler) -> None:
        """Install (or replace) the incoming request handler."""
        self._request_handler = handler

    def on_notification(self, handler: NotificationHandler) -> None:
        """Install (or replace) the incoming notification handler."""
        self._notification_handler = handler

    def start(self) -> None:
        """Attach the read loop once; close is permanent, never re-attached."""
        if self._started:
            return
        self._started = True
        self._task = asyncio.create_task(self._read_loop())

    async def close(self) -> None:
        """Stop reading and reject pending requests without closing the streams."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._fail_pending(RuntimeError("JSON-RPC transport closed"))

    async def flush(self) -> None:
        """Order prior frames behind an empty barrier and surface write failures."""
        self._writer.write(b"")
        await self._writer.drain()

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a notification, omitting ``params`` when none were supplied."""
        frame: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            frame["params"] = params
        self._write(frame)

    async def request(
        self,
        method: str,
        params: dict[str, Any],
        signal: AbortSignal | None = None,
    ) -> Any:
        """Send a request and await its response, rejecting error frames."""
        if self._failure is not None:
            raise self._failure
        if signal is not None and signal.is_aborted:
            raise _abort_error(signal.reason)
        request_id = f"req_{uuid.uuid4().hex}"
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future

        def _abort() -> None:
            pending = self._pending.pop(request_id, None)
            if pending is not None and not pending.done():
                pending.set_exception(
                    _abort_error(signal.reason if signal is not None else None)
                )

        detach = signal.on_abort(_abort) if signal is not None else (lambda: None)
        try:
            self._write(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            )
        except Exception:
            self._pending.pop(request_id, None)
            detach()
            raise
        try:
            return await future
        finally:
            detach()
            self._pending.pop(request_id, None)

    def _write(self, frame: dict[str, Any]) -> None:
        line = json.dumps(frame, separators=(",", ":"), ensure_ascii=False)
        self._writer.write((line + "\n").encode("utf-8"))

    async def _read_loop(self) -> None:
        try:
            while True:
                raw = await self._reader.readline()
                if raw == b"":
                    self._fail_pending(RuntimeError("JSON-RPC input closed"))
                    return
                await self._handle_line(raw.decode("utf-8", errors="replace").strip())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - transport boundary
            self._fail_pending(exc)

    async def _handle_line(self, line: str) -> None:
        if not line:
            return
        try:
            frame = json.loads(line)
        except (json.JSONDecodeError, UnicodeError):
            return
        if not isinstance(frame, dict):
            return
        frame_id, method = frame.get("id"), frame.get("method")
        if _valid_id(frame_id) and isinstance(method, str):
            identifier = cast(JsonRpcId, frame_id)
            await self._handle_request(
                identifier, method, _object_params(frame.get("params"))
            )
        elif _valid_id(frame_id):
            self._handle_response(cast(JsonRpcId, frame_id), frame)
        elif isinstance(method, str):
            handler = self._notification_handler
            if handler is not None:
                handler(method, _object_params(frame.get("params")))

    async def _handle_request(
        self, request_id: JsonRpcId, method: str, params: dict[str, Any]
    ) -> None:
        handler = self._request_handler
        if handler is None:
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": METHOD_NOT_FOUND,
                        "message": f"method not found: {method}",
                    },
                }
            )
            return
        try:
            result = await handler(method, params)
        except Exception as exc:  # noqa: BLE001 - handler boundary
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": INTERNAL_ERROR, "message": str(exc)},
                }
            )
        else:
            self._write({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _handle_response(self, request_id: JsonRpcId, frame: dict[str, Any]) -> None:
        pending = self._pending.pop(request_id, None)
        if pending is None or pending.done():
            return
        error = frame.get("error")
        if isinstance(error, (dict, list)):
            code = error.get("code") if isinstance(error, dict) else None
            message = error.get("message") if isinstance(error, dict) else None
            data = error.get("data") if isinstance(error, dict) else None
            pending.set_exception(
                JsonRpcResponseError(
                    code
                    if isinstance(code, (int, float)) and not isinstance(code, bool)
                    else None,
                    message if isinstance(message, str) else "JSON-RPC error",
                    data,
                )
            )
            return
        pending.set_result(frame.get("result"))

    def _fail_pending(self, error: Exception) -> None:
        if self._failure is None:
            self._failure = error
        for pending in list(self._pending.values()):
            if not pending.done():
                pending.set_exception(error)
        self._pending.clear()
