"""Shared subprocess-backed shell executor mechanics for bash and pwsh providers."""

from __future__ import annotations

import asyncio
import math
import os
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, ClassVar

from anima import Context

from magitech_harness.agent.cancellation import (
    MAX_TIMER_DELAY_MS,
    AbortController,
    AbortSignal,
    clamp_timeout,
    deadline,
    timeout_of,
)
from magitech_harness.shell.executor import ShellExecutor
from magitech_harness.shell.types import (
    CollectedOutput,
    CollectedOutputReader,
    OutputReadSlice,
    ShellExecRequest,
    ShellExecSpec,
    ShellExecution,
    ShellObservedStreams,
    ShellProcessRead,
    ShellProcessStatus,
    ShellRunResult,
    ShellSandboxInfo,
)
from magitech_harness.subprocess.types import (
    SubprocessCollect,
    SubprocessHandle,
    SubprocessSpawnSpec,
    SubprocessStdinMode,
)

DEFAULT_TIMEOUT_MS = 120_000
DEFAULT_MAX_TIMEOUT_MS = 600_000
DEFAULT_MAX_OUTPUT_BYTES = 64_000
DEFAULT_MAX_SPILL_BYTES = 64 * 1024 * 1024
DEFAULT_GRACE_MS = 3_000
_SHELL_TIMEOUT_CODE = "BASH_TIMEOUT"


def assert_positive_finite(prefix: str, name: str, value: Any) -> float:
    """Validate that *value* is a positive finite int or float (rejecting bool)."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{prefix}: {name} must be a positive finite number")
    return float(value)


def assert_serviceable_config(prefix: str, config: Mapping[str, Any]) -> None:
    """Validate live budget fields required before every shell resolution."""
    assert_positive_finite(prefix, "timeoutMs", config["timeoutMs"])
    assert_positive_finite(prefix, "maxTimeoutMs", config["maxTimeoutMs"])
    assert_positive_finite(prefix, "maxOutputBytes", config["maxOutputBytes"])
    assert_positive_finite(prefix, "maxSpillBytes", config["maxSpillBytes"])
    grace = assert_positive_finite(prefix, "graceMs", config["graceMs"])
    if grace > MAX_TIMER_DELAY_MS:
        raise ValueError(
            f"{prefix}: graceMs must be no greater than {MAX_TIMER_DELAY_MS}"
        )


def _final_output(reader: CollectedOutputReader) -> CollectedOutput:
    read = reader.read_from(0)
    out: CollectedOutput = {"text": read["text"], "truncated": read["lossy"]}
    if "spillPath" in read:
        out["spillPath"] = read["spillPath"]
    return out


class _EmptyReader:
    """Reader returning empty slices when no subprocess was launched."""

    def read_from(self, from_byte: int) -> OutputReadSlice:
        _ = from_byte
        return {"text": "", "nextOffset": 0, "lossy": False}

    def snapshot(self) -> tuple[bytes, int]:  # pragma: no cover - protocol parity
        return b"", 0

    def seal(self) -> None:  # pragma: no cover - protocol parity
        return None

    def finalize(self) -> CollectedOutput:  # pragma: no cover - protocol parity
        return {"text": "", "truncated": False}


class _ObservedStderrReader:
    """Observed stderr reader serving provider failures when spawn failed."""

    def __init__(
        self,
        delegate: CollectedOutputReader,
        get_failure_note: Callable[[], str | None],
    ) -> None:
        self._delegate = delegate
        self._get_failure_note = get_failure_note

    def read_from(self, from_byte: int) -> OutputReadSlice:
        note = self._get_failure_note()
        if note is None:
            return self._delegate.read_from(from_byte)
        raw = note.encode("utf-8")
        offset = min(max(0, from_byte), len(raw))
        return {
            "text": raw[offset:].decode("utf-8", errors="replace"),
            "nextOffset": len(raw),
            "lossy": False,
        }

    def snapshot(self) -> tuple[bytes, int]:  # pragma: no cover - protocol parity
        return self._delegate.snapshot()

    def seal(self) -> None:  # pragma: no cover - protocol parity
        self._delegate.seal()

    def finalize(self) -> CollectedOutput:  # pragma: no cover - protocol parity
        return self._delegate.finalize()


class _SubprocessShellExecution:
    """Concrete :class:`ShellExecution` handle wrapping subprocess execution."""

    def __init__(
        self,
        *,
        stdout_reader: CollectedOutputReader,
        stderr_reader: CollectedOutputReader,
        observed_stderr: CollectedOutputReader,
        get_failure: Callable[[], tuple[BaseException, str] | None],
        consume_failure: Callable[[], str],
        on_kill: Callable[[], None],
        classify: Callable[[], tuple[bool, bool]],
        timeout_ms: float,
        done: asyncio.Future[None],
    ) -> None:
        self.status: ShellProcessStatus = "running"
        self.exit_code: int | None = None
        self.signal: str | None = None
        self.sandbox: ShellSandboxInfo | None = None
        self._stdout_reader = stdout_reader
        self._stderr_reader = stderr_reader
        self._observed: ShellObservedStreams = {
            "stdout": stdout_reader,
            "stderr": observed_stderr,
        }
        self._stdout_offset = 0
        self._stderr_offset = 0
        self._get_failure = get_failure
        self._consume_failure = consume_failure
        self._on_kill = on_kill
        self._classify = classify
        self._timeout_ms = timeout_ms
        self._done = done
        self._result_future: asyncio.Future[ShellRunResult] | None = None

    @property
    def done(self) -> asyncio.Future[None]:
        return self._done

    @property
    def observed(self) -> ShellObservedStreams:
        return self._observed

    def read_output(self) -> ShellProcessRead:
        out = self._stdout_reader.read_from(self._stdout_offset)
        err = self._stderr_reader.read_from(self._stderr_offset)
        self._stdout_offset = out["nextOffset"]
        self._stderr_offset = err["nextOffset"]

        failure_note = self._consume_failure()
        failure_sep = "\n" if err["text"] and not err["text"].endswith("\n") else ""
        err_text = (
            f"{err['text']}{failure_sep}{failure_note}" if failure_note else err["text"]
        )
        sep = "\n" if out["text"] and not out["text"].endswith("\n") else ""
        delta = f"{out['text']}{sep}[stderr]\n{err_text}" if err_text else out["text"]

        payload: ShellProcessRead = {
            "delta": delta,
            "lossy": bool(out["lossy"] or err["lossy"]),
        }
        if "spillPath" in out:
            payload["stdoutSpillPath"] = out["spillPath"]
        if "spillPath" in err:
            payload["stderrSpillPath"] = err["spillPath"]
        return payload

    def kill(self) -> bool:
        if self.status != "running":
            return False
        self.status = "killed"
        self._on_kill()
        return True

    def result(self) -> asyncio.Future[ShellRunResult]:
        if self._result_future is None:
            loop = asyncio.get_running_loop()

            async def _project() -> ShellRunResult:
                await self._done
                failure = self._get_failure()
                if failure is not None:
                    raise failure[0]
                timed_out, aborted = self._classify()
                res: ShellRunResult = {
                    "exitCode": self.exit_code,
                    "signal": self.signal,
                    "timedOut": timed_out,
                    "aborted": aborted,
                    "timeoutMs": self._timeout_ms,
                    "stdout": _final_output(self._stdout_reader),
                    "stderr": _final_output(self._stderr_reader),
                }
                if self.sandbox is not None:
                    res["sandbox"] = self.sandbox
                return res

            self._result_future = loop.create_task(_project())
        return self._result_future


class SubprocessShellExecutor(ShellExecutor, ABC):
    """Shared base for subprocess-backed local shell executors."""

    inject: ClassVar[list[str]] = ["subprocess"]
    _config_prefix: ClassVar[str] = "shell-local"
    _env_overrides: ClassVar[Mapping[str, str]] = {}

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx)
        raw = dict(config or {})
        self.config: dict[str, Any] = {
            "cwd": raw.get("cwd"),
            "timeoutMs": raw.get("timeoutMs", DEFAULT_TIMEOUT_MS),
            "maxTimeoutMs": raw.get("maxTimeoutMs", DEFAULT_MAX_TIMEOUT_MS),
            "maxOutputBytes": raw.get("maxOutputBytes", DEFAULT_MAX_OUTPUT_BYTES),
            "maxSpillBytes": raw.get("maxSpillBytes", DEFAULT_MAX_SPILL_BYTES),
            "graceMs": raw.get("graceMs", DEFAULT_GRACE_MS),
        }
        if "pwshPath" in raw:
            self.config["pwshPath"] = raw["pwshPath"]
        self._settle_tasks: set[asyncio.Task[None]] = set()

    def resolve(self, request: ShellExecRequest) -> ShellExecSpec:
        """Apply defaults and clamps to produce a resolved :class:`ShellExecSpec`."""
        assert_serviceable_config(self._config_prefix, self.config)
        timeout_ms = clamp_timeout(
            request.get("timeoutMs"),
            float(self.config["timeoutMs"]),
            float(self.config["maxTimeoutMs"]),
            f"{self._config_prefix}: request.timeoutMs",
        )
        raw_stdout_max = request.get("stdoutMaxBytes")
        stdout_max = (
            int(self.config["maxOutputBytes"])
            if raw_stdout_max is None
            else int(
                assert_positive_finite(
                    "request.stdoutMaxBytes", "stdoutMaxBytes", raw_stdout_max
                )
            )
        )
        workdir = request.get("workdir") or self.config.get("cwd") or os.getcwd()
        on_expiry = request.get("onExpiry") or "kill"
        spec: ShellExecSpec = {
            "command": request["command"],
            "workdir": workdir,
            "timeoutMs": timeout_ms,
            "onExpiry": on_expiry,
            "stdoutMaxBytes": stdout_max,
            "sandboxPolicy": request.get("sandboxPolicy"),
        }
        if request.get("signal") is not None:
            sig = request["signal"]
            assert sig is not None
            spec["signal"] = sig
        if request.get("stdin") is not None:
            stdin_val = request["stdin"]
            assert stdin_val is not None
            spec["stdin"] = stdin_val
        if request.get("env") is not None:
            env_val = request["env"]
            assert env_val is not None
            spec["env"] = env_val
        if request.get("magitechEnv") is not None:
            magitech_env = request["magitechEnv"]
            assert magitech_env is not None
            spec["magitechEnv"] = magitech_env
        return spec

    def _spawn_spec(
        self,
        spec: ShellExecSpec,
        argv: Sequence[str],
        stdout_max_bytes: int,
        signal: AbortSignal | None,
    ) -> SubprocessSpawnSpec:
        max_spill = int(self.config["maxSpillBytes"])

        def _collect(max_bytes: int) -> SubprocessCollect:
            return {
                "kind": "collect",
                "maxBytes": max_bytes,
                "spill": {"maxBytes": max_spill},
            }

        stdin_mode: SubprocessStdinMode = (
            {"kind": "bytes", "data": spec["stdin"].encode("utf-8")}
            if "stdin" in spec
            else "ignore"
        )
        env: dict[str, str] = dict(self._env_overrides)
        if "env" in spec:
            env.update(spec["env"])
        if "magitechEnv" in spec:
            env.update(spec["magitechEnv"])

        return {
            "argv": list(argv),
            "cwd": spec["workdir"],
            "stdio": {
                "stdin": stdin_mode,
                "stdout": _collect(stdout_max_bytes),
                "stderr": _collect(int(self.config["maxOutputBytes"])),
            },
            "graceMs": float(self.config["graceMs"]),
            "signal": signal,
            "env": env,
        }

    async def execute_argv(
        self,
        spec: ShellExecSpec,
        argv_or_prepare: Sequence[str]
        | Callable[[AbortSignal], Awaitable[Sequence[str]]],
        on_started: Callable[[ShellExecution], None] | None = None,
    ) -> ShellExecution:
        """Prepare argv, launch via ``ctx.subprocess``, and publish handle."""
        if spec["onExpiry"] == "kill":
            dl = deadline(spec.get("signal"), spec["timeoutMs"], _SHELL_TIMEOUT_CODE)
            spawn_signal: AbortSignal | None = dl.signal

            def _classify() -> tuple[bool, bool]:
                timed_out = timeout_of(dl.signal, _SHELL_TIMEOUT_CODE) is not None
                return timed_out, dl.signal.aborted and not timed_out

            disarm: Callable[[], None] = dl.dispose
        else:
            spawn_signal = spec.get("signal")

            def _classify() -> tuple[bool, bool]:
                return False, bool(spawn_signal is not None and spawn_signal.aborted)

            disarm = lambda: None

        argv: Sequence[str] = ()
        preparation_timed_out = False
        if callable(argv_or_prepare):
            prep_signal = spawn_signal or AbortController().signal
            loop = asyncio.get_running_loop()
            cancelled: asyncio.Future[Sequence[str]] = loop.create_future()

            def _on_prep_abort() -> None:
                err = (
                    prep_signal.reason
                    if isinstance(prep_signal.reason, BaseException)
                    else RuntimeError("aborted")
                )
                cancelled.set_exception(err)

            unsub = prep_signal.on_abort(_on_prep_abort)

            async def _run_prepare() -> Sequence[str]:
                return await argv_or_prepare(prep_signal)

            prep_task: asyncio.Task[Sequence[str]] = loop.create_task(_run_prepare())
            try:
                await asyncio.wait(
                    (prep_task, cancelled),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancelled.done():
                    prep_task.cancel()
                    await asyncio.gather(prep_task, return_exceptions=True)
                    argv = await cancelled
                else:
                    argv = await prep_task
            except BaseException:
                if not _classify()[0]:
                    disarm()
                    raise
                preparation_timed_out = True
            finally:
                unsub()
        else:
            argv = argv_or_prepare

        running: SubprocessHandle | None = None
        sync_spawn_error: BaseException | None = None
        if not preparation_timed_out:
            try:
                running = self.ctx.subprocess.spawn(
                    self._spawn_spec(spec, argv, spec["stdoutMaxBytes"], spawn_signal)
                )
            except BaseException as exc:  # noqa: BLE001 - normalize sync & async spawn errors
                sync_spawn_error = exc

        if running is not None:
            collected = running.collected
            stdout_reader = collected["stdout"]
            stderr_reader = collected["stderr"]
        else:
            stdout_reader = _EmptyReader()
            stderr_reader = _EmptyReader()

        provider_failure: tuple[BaseException, str] | None = None
        failure_reported = False

        def _get_failure() -> tuple[BaseException, str] | None:
            return provider_failure

        def _get_failure_note() -> str | None:
            return None if provider_failure is None else provider_failure[1]

        def _consume_failure() -> str:
            nonlocal failure_reported
            if provider_failure is None or failure_reported:
                return ""
            failure_reported = True
            return provider_failure[1]

        observed_stderr: CollectedOutputReader = _ObservedStderrReader(
            stderr_reader, _get_failure_note
        )
        loop = asyncio.get_running_loop()
        done: asyncio.Future[None] = loop.create_future()

        def _terminate() -> None:
            if running is not None:
                running.terminate()

        proc = _SubprocessShellExecution(
            stdout_reader=stdout_reader,
            stderr_reader=stderr_reader,
            observed_stderr=observed_stderr,
            get_failure=_get_failure,
            consume_failure=_consume_failure,
            on_kill=_terminate,
            classify=_classify,
            timeout_ms=spec["timeoutMs"],
            done=done,
        )

        async def _settle_handle() -> None:
            nonlocal provider_failure
            try:
                if preparation_timed_out:
                    proc.status = "killed"
                    self.on_process_done(proc, "", False, None)
                    return
                if sync_spawn_error is not None:
                    provider_failure = (
                        sync_spawn_error,
                        f"subprocess failed before reporting an outcome: {sync_spawn_error}",
                    )
                    proc.status = "killed"
                    self.on_process_done(
                        proc, provider_failure[1], True, sync_spawn_error
                    )
                    return
                assert running is not None
                try:
                    outcome = await running.done
                except BaseException as exc:  # noqa: BLE001 - contain provider rejection
                    if proc.status == "killed" or (
                        spawn_signal is not None and spawn_signal.aborted
                    ):
                        proc.status = "killed"
                        self.on_process_done(
                            proc, stderr_reader.read_from(0)["text"], False, None
                        )
                    else:
                        provider_failure = (
                            exc,
                            f"subprocess failed before reporting an outcome: {exc}",
                        )
                        proc.status = "killed"
                        self.on_process_done(proc, provider_failure[1], True, exc)
                    return

                if proc.status == "running":
                    proc.status = (
                        "killed"
                        if (spawn_signal is not None and spawn_signal.aborted)
                        or outcome["signal"] is not None
                        else "completed"
                    )
                proc.exit_code = outcome["exitCode"]
                proc.signal = outcome["signal"]
                self.on_process_done(
                    proc, stderr_reader.read_from(0)["text"], False, None
                )
            finally:
                disarm()
                done.set_result(None)

        settle_task = loop.create_task(_settle_handle())
        self._settle_tasks.add(settle_task)
        settle_task.add_done_callback(self._settle_tasks.discard)
        if not preparation_timed_out and on_started is not None:
            on_started(proc)
        return proc

    def on_process_done(
        self,
        proc: ShellExecution,
        stderr: str,
        provider_rejected: bool,
        provider_error: BaseException | None = None,
    ) -> None:
        """Hook called right before ``proc.done`` settles."""
        _ = (proc, stderr, provider_rejected, provider_error)

    @abstractmethod
    async def execute(self, spec: ShellExecSpec) -> ShellExecution:
        """Start command for *spec*."""


__all__ = [
    "DEFAULT_GRACE_MS",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_MAX_SPILL_BYTES",
    "DEFAULT_MAX_TIMEOUT_MS",
    "DEFAULT_TIMEOUT_MS",
    "SubprocessShellExecutor",
    "assert_positive_finite",
    "assert_serviceable_config",
]
