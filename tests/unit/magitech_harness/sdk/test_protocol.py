"""Behavioral probes for the SDK wire protocol port (row 104)."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest

from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.sdk import protocol


class _Writer:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.chunks.append(data)

    async def drain(self) -> None:
        return None

    def frames(self) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for chunk in self.chunks
            for line in chunk.decode().splitlines()
        ]


def _transport() -> tuple[asyncio.StreamReader, _Writer, protocol.JsonRpcLineTransport]:
    reader = asyncio.StreamReader()
    writer = _Writer()
    transport = protocol.JsonRpcLineTransport(
        cast(asyncio.StreamReader, reader), cast(asyncio.StreamWriter, writer)
    )
    transport.start()
    return reader, writer, transport


def _feed(reader: asyncio.StreamReader, *frames: Any) -> None:
    for frame in frames:
        text = (
            frame
            if isinstance(frame, str)
            else json.dumps(frame, separators=(",", ":"))
        )
        reader.feed_data((text + "\n").encode())


async def _tick() -> None:
    await asyncio.sleep(0)


def test_method_vocabulary_is_stable() -> None:
    assert protocol.CLIENT_REQUEST_METHODS[0] == "initialize"
    assert protocol.HOST_REQUEST_METHODS == ("approval/request", "user-question/ask")
    assert len(protocol.NOTIFICATION_METHODS) == 6
    assert set(protocol.HarnessSdkClientRequestMap) == set(
        protocol.CLIENT_REQUEST_METHODS
    )
    assert set(protocol.HarnessSdkHostRequestMap) == set(protocol.HOST_REQUEST_METHODS)
    assert set(protocol.HarnessSdkNotificationMap) == set(protocol.NOTIFICATION_METHODS)
    assert protocol.HarnessSdkRequestMap["initialize"][1] is protocol.InitializeResult


@pytest.mark.asyncio
async def test_notify_and_flush_emit_one_frame_per_line() -> None:
    _, writer, transport = _transport()
    transport.notify("session.status", {"sessionId": "s", "status": "idle"})
    transport.notify("shutdown")
    await transport.flush()
    assert writer.frames() == [
        {
            "jsonrpc": "2.0",
            "method": "session.status",
            "params": {"sessionId": "s", "status": "idle"},
        },
        {"jsonrpc": "2.0", "method": "shutdown"},
    ]
    await transport.close()


@pytest.mark.asyncio
async def test_request_response_and_error_semantics() -> None:
    reader, _, transport = _transport()
    _feed(reader, {"jsonrpc": "2.0", "id": "pending", "result": {"ok": True}})
    _feed(reader, {"jsonrpc": "2.0", "id": "unknown", "result": None})
    await _tick()

    task = asyncio.create_task(transport.request("session/wait", {"sessionId": "s"}))
    await _tick()
    request_id = next(iter(transport._pending))
    _feed(reader, {"jsonrpc": "2.0", "id": request_id, "result": "done"})
    assert await task == "done"
    assert transport._pending == {}

    error_task = asyncio.create_task(transport.request("initialize", {}))
    await _tick()
    error_id = next(iter(transport._pending))
    _feed(
        reader,
        {
            "jsonrpc": "2.0",
            "id": error_id,
            "error": {"code": -32603, "message": "bad", "data": {"x": 1}},
        },
    )
    with pytest.raises(protocol.JsonRpcResponseError) as caught:
        await error_task
    assert caught.value.code == -32603
    assert caught.value.data == {"x": 1}
    assert str(caught.value) == "bad"

    malformed_task = asyncio.create_task(transport.request("initialize", {}))
    await _tick()
    malformed_id = next(iter(transport._pending))
    _feed(
        reader,
        {"jsonrpc": "2.0", "id": malformed_id, "error": {"code": "x", "message": 7}},
    )
    with pytest.raises(protocol.JsonRpcResponseError) as fallback:
        await malformed_task
    assert fallback.value.code is None
    assert str(fallback.value) == "JSON-RPC error"
    await transport.close()


@pytest.mark.asyncio
async def test_incoming_requests_and_notifications() -> None:
    reader, writer, transport = _transport()
    seen: list[tuple[str, dict[str, Any]]] = []

    async def _handle(method: str, params: dict[str, Any]) -> Any:
        seen.append((method, params))
        if method == "boom":
            raise RuntimeError("failed")
        return {"echo": method}

    transport.on_request(_handle)
    transport.on_notification(lambda method, params: seen.append((method, params)))
    _feed(
        reader,
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"a": 1}},
        {"jsonrpc": "2.0", "id": 2, "method": "boom", "params": []},
        {"jsonrpc": "2.0", "method": "session.status", "params": {"status": "idle"}},
        {"jsonrpc": "2.0", "method": "shutdown"},
        {"jsonrpc": "2.0", "id": 1.5, "method": "fractional", "params": {}},
        "not json",
        "[]",
        "",
    )
    await _tick()
    assert seen[:5] == [
        ("initialize", {"a": 1}),
        ("boom", {}),
        ("session.status", {"status": "idle"}),
        ("shutdown", {}),
        ("fractional", {}),
    ]
    frames = writer.frames()
    assert frames[0]["result"] == {"echo": "initialize"}
    assert frames[1]["error"] == {"code": protocol.INTERNAL_ERROR, "message": "failed"}
    assert frames[2] == {"jsonrpc": "2.0", "id": 1.5, "result": {"echo": "fractional"}}
    await transport.close()


@pytest.mark.asyncio
async def test_missing_request_handler_and_ignored_notifications() -> None:
    reader, writer, transport = _transport()
    _feed(
        reader,
        {"jsonrpc": "2.0", "id": 9, "method": "unknown", "params": 7},
        {"jsonrpc": "2.0", "method": "dropped"},
    )
    await _tick()
    assert writer.frames() == [
        {
            "jsonrpc": "2.0",
            "id": 9,
            "error": {
                "code": protocol.METHOD_NOT_FOUND,
                "message": "method not found: unknown",
            },
        }
    ]
    await transport.close()


@pytest.mark.asyncio
async def test_close_and_input_end_reject_pending_requests() -> None:
    _, _, transport = _transport()
    pending = asyncio.create_task(transport.request("shutdown", {}))
    await _tick()
    await transport.close()
    with pytest.raises(RuntimeError, match="transport closed"):
        await pending
    with pytest.raises(RuntimeError, match="transport closed"):
        await transport.request("shutdown", {})
    transport.start()  # start is a permanent no-op after close
    assert transport._started is True
    assert transport._task is None
    with pytest.raises(RuntimeError, match="transport closed"):
        await transport.request("shutdown", {})

    reader2, _, transport2 = _transport()
    pending2 = asyncio.create_task(transport2.request("shutdown", {}))
    await _tick()
    reader2.feed_eof()
    await _tick()
    with pytest.raises(RuntimeError, match="input closed"):
        await pending2
    await transport2.close()


@pytest.mark.asyncio
async def test_abort_signal_rejects_and_disposes_pending_requests() -> None:
    reader, _, transport = _transport()
    signal = AbortSignal()
    signal.abort("early")
    with pytest.raises(AbortError):
        await transport.request("shutdown", {}, signal)

    live = AbortSignal()
    task = asyncio.create_task(transport.request("shutdown", {}, live))
    await _tick()
    request_id = next(iter(transport._pending))
    live.abort("user")
    with pytest.raises(AbortError):
        await task
    assert transport._pending == {}
    _feed(reader, {"jsonrpc": "2.0", "id": request_id, "result": "late"})
    await _tick()
    await transport.close()


@pytest.mark.asyncio
async def test_read_loop_failures_reject_pending_requests() -> None:
    class _FailingReader:
        async def readline(self) -> bytes:
            raise OSError("read failed")

    transport = protocol.JsonRpcLineTransport(
        cast(asyncio.StreamReader, _FailingReader()),
        cast(asyncio.StreamWriter, _Writer()),
    )
    transport.start()
    await _tick()
    with pytest.raises(OSError, match="read failed"):
        await transport.request("shutdown", {})
    await transport.close()


def test_object_params_and_valid_id_helpers() -> None:
    assert protocol._object_params({"a": 1}) == {"a": 1}
    assert protocol._object_params([1]) == {}
    assert protocol._object_params("s") == {}
    assert protocol._valid_id("id") is True
    assert protocol._valid_id(3) is True
    assert protocol._valid_id(1.5) is True
    assert protocol._valid_id(True) is False
    assert protocol._valid_id(None) is False


@pytest.mark.asyncio
async def test_idempotent_lifecycle_and_defensive_branches() -> None:
    _, _, transport = _transport()
    transport.start()  # idempotent: a second start must not replace the task
    await transport.close()
    await transport.close()  # closing without a task is a no-op

    signal = AbortSignal()
    task = asyncio.create_task(transport.request("shutdown", {}, signal))
    # The transport is already closed, so this request fails immediately.
    with pytest.raises(RuntimeError):
        await task

    reader2, _, transport2 = _transport()
    signal2 = AbortSignal()
    pending = asyncio.create_task(transport2.request("shutdown", {}, signal2))
    await _tick()
    listeners = list(signal2._listeners)
    request_id = next(iter(transport2._pending))
    _feed(reader2, {"jsonrpc": "2.0", "id": request_id, "result": "ok"})
    assert await pending == "ok"
    listeners[0]()  # late abort callback after the pending entry was removed

    _feed(reader2, {"jsonrpc": "2.0"}, {"jsonrpc": "2.0", "id": [], "method": 7})
    await _tick()

    done: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    done.set_result("already done")
    transport2._pending["done"] = done
    transport2._fail_pending(RuntimeError("defensive"))
    assert transport2._pending == {}
    await transport2.close()


@pytest.mark.asyncio
async def test_flush_barrier_and_failure() -> None:
    _, writer, transport = _transport()
    await transport.flush()
    assert writer.chunks[-1] == b""

    class _FailBarrier:
        def write(self, _data: bytes) -> None:
            raise OSError("flush failed")

        async def drain(self) -> None:
            return None

    failing = protocol.JsonRpcLineTransport(
        cast(asyncio.StreamReader, asyncio.StreamReader()),
        cast(asyncio.StreamWriter, _FailBarrier()),
    )
    with pytest.raises(OSError, match="flush failed"):
        await failing.flush()


@pytest.mark.asyncio
async def test_request_write_failure_releases_pending_state() -> None:
    class _FailOnceWriter(_Writer):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True

        def write(self, data: bytes) -> None:
            if self.fail:
                self.fail = False
                raise OSError("write exploded")
            super().write(data)

    reader = asyncio.StreamReader()
    writer = _FailOnceWriter()
    transport = protocol.JsonRpcLineTransport(
        cast(asyncio.StreamReader, reader), cast(asyncio.StreamWriter, writer)
    )
    transport.start()
    with pytest.raises(OSError, match="write exploded"):
        await transport.request("shutdown", {})
    assert transport._pending == {}

    pending = asyncio.create_task(transport.request("shutdown", {}))
    await _tick()
    request_id = next(iter(transport._pending))
    _feed(reader, {"jsonrpc": "2.0", "id": request_id, "result": "recovered"})
    assert await pending == "recovered"
    await transport.close()


@pytest.mark.asyncio
async def test_array_error_frame_and_abort_reason_shapes() -> None:
    reader, _, transport = _transport()
    pending = asyncio.create_task(transport.request("initialize", {}))
    await _tick()
    request_id = next(iter(transport._pending))
    _feed(reader, {"jsonrpc": "2.0", "id": request_id, "error": []})
    with pytest.raises(protocol.JsonRpcResponseError) as caught:
        await pending
    assert caught.value.code is None
    assert str(caught.value) == "JSON-RPC error"

    error_signal = AbortSignal()
    error_signal.abort(RuntimeError("already gone"))
    with pytest.raises(AbortError) as error_abort:
        await transport.request("shutdown", {}, error_signal)
    assert str(error_abort.value) == "already gone"

    text_signal = AbortSignal()
    text_signal.abort("plain reason")
    with pytest.raises(AbortError) as text_abort:
        await transport.request("shutdown", {}, text_signal)
    assert str(text_abort.value) == "JSON-RPC request aborted: plain reason"
    await transport.close()


@pytest.mark.asyncio
async def test_multibyte_utf8_split_across_chunks() -> None:
    reader, _, transport = _transport()
    seen: list[tuple[str, dict[str, Any]]] = []
    transport.on_notification(lambda method, params: seen.append((method, params)))
    frame = json.dumps(
        {"jsonrpc": "2.0", "method": "text", "params": {"value": "é✓"}},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    split = frame.index("é".encode()) + 1
    reader.feed_data(frame[:split])
    await _tick()
    reader.feed_data(frame[split:] + b"\n")
    await _tick()
    assert seen == [("text", {"value": "é✓"})]
    await transport.close()


def test_wire_payload_fields_match_reference() -> None:
    assert {"cwd", "provider", "model"} <= set(
        protocol.InitializeParams.__required_keys__
    )
    assert "timeZone" not in protocol.InitializeParams.__annotations__
    assert set(protocol.InitializeResult.__required_keys__) == {"serverInfo"}
    assert "path" in protocol.SessionWorkingDirectorySetParams.__annotations__
    assert "cwd" not in protocol.SessionWorkingDirectorySetParams.__annotations__
    assert set(protocol.SdkUserQuestionCancelled.__required_keys__) == {
        "requestId",
        "sessionId",
    }
    assert set(protocol.SubagentStartedNotification.__annotations__) == {
        "parentSessionId",
        "childSessionId",
    }
    assert "parentSessionId" in protocol.SubagentFinishedNotification.__annotations__
    assert set(protocol.SessionPromptParams.__required_keys__) == {
        "sessionId",
        "contentBlocks",
    }
    assert set(protocol.SessionQueryParams.__required_keys__) == {
        "sessionId",
        "contentBlocks",
    }
    assert protocol.SessionResumeParams.__required_keys__ == {"sessionId"}
    assert set(protocol.ApprovalRequestParams.__required_keys__) == {
        "sessionId",
        "toolName",
    }
    assert set(protocol.SdkUserQuestionRequest.__required_keys__) == {
        "requestId",
        "sessionId",
        "questions",
    }
    assert set(protocol.SessionEventNotification.__required_keys__) == {
        "sessionId",
        "event",
    }
    assert set(protocol.SessionStatusNotification.__required_keys__) == {
        "sessionId",
        "status",
    }
    assert set(protocol.SessionQueryCompletedNotification.__required_keys__) == {
        "sessionId",
        "commandId",
        "finalResponse",
        "finishReason",
    }
