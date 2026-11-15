"""Unit tests for tools contracts and structured errors."""

from __future__ import annotations

from magitech_harness.llm.error import HarnessError
from magitech_harness.tools.contracts import (
    RUN_CODE_NAME,
    TOOL_ABORTED,
    TOOL_ABORTED_BEFORE_DISPATCH,
    ToolActivationError,
    ToolBatchExecutionError,
    ToolBatchResult,
    ToolExecutorUnavailableError,
    ToolNotFoundError,
    ToolOutputError,
)


def test_tool_error_classes_and_constants() -> None:
    """Stable constants, HarnessError codes, causes, and partial batch payloads."""
    assert RUN_CODE_NAME == "run_code"
    assert TOOL_ABORTED == "ABORTED"
    assert TOOL_ABORTED_BEFORE_DISPATCH == "ABORTED_BEFORE_DISPATCH"

    not_found = ToolNotFoundError("ghost")
    assert isinstance(not_found, HarnessError)
    assert not_found.code == "UNKNOWN_TOOL"
    assert str(not_found) == 'unknown tool "ghost"'

    not_found_reachable = ToolNotFoundError("ghost", "call via run_code")
    assert str(not_found_reachable) == 'unknown tool "ghost": call via run_code'

    cause = RuntimeError("vault down")
    act_err = ToolActivationError("search", cause=cause)
    assert isinstance(act_err, HarnessError)
    assert act_err.code == "TOOL_ACTIVATION_FAILED"
    assert str(act_err) == 'cannot resolve availability of tool "search"'
    assert act_err.cause is cause

    out_err = ToolOutputError("my_tool", ['"value" must be a string'])
    assert isinstance(out_err, HarnessError)
    assert out_err.code == "INVALID_TOOL_OUTPUT"
    assert out_err.violations == ['"value" must be a string']
    assert (
        str(out_err)
        == 'tool "my_tool" returned invalid output: "value" must be a string'
    )

    str_out_err = ToolOutputError("my_tool", "non-lossless payload")
    assert str_out_err.violations == ["non-lossless payload"]
    assert "non-lossless payload" in str(str_out_err)

    unavail = ToolExecutorUnavailableError()
    assert isinstance(unavail, HarnessError)
    assert unavail.code == "TOOL_EXECUTOR_UNAVAILABLE"
    assert "unavailable" in str(unavail)

    partial: ToolBatchResult = {
        "results": [
            {"callId": "c1", "result": {"isError": False, "value": 1, "content": []}}
        ],
        "additionalContexts": [],
        "concluded": False,
    }
    batch_err = ToolBatchExecutionError(cause, partial)
    assert batch_err.partial is partial
    assert batch_err.cause is cause
    assert batch_err.__cause__ is cause

    non_exc_err = ToolBatchExecutionError("plain string", partial)
    assert non_exc_err.cause == "plain string"
    assert non_exc_err.__cause__ is None
