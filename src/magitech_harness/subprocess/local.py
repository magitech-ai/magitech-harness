"""Local host SubprocessRuntime implementation (LocalSubprocessRuntime)."""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import signal as signal_mod
import subprocess as py_subprocess
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import IO, Any, cast

from anima import Context

from magitech_harness.agent.cancellation import MAX_TIMER_DELAY_MS, AbortSignal
from magitech_harness.subprocess.output import (
    OutputCollector,
    SpillOptions,
    default_spill_dir,
    log_spill_failure,
)
from magitech_harness.subprocess.service import (
    SubprocessExecutableNotFoundError,
    SubprocessRuntime,
    child_env,
)
from magitech_harness.subprocess.types import (
    SubprocessCollect,
    SubprocessCollected,
    SubprocessOutcome,
    SubprocessOutputMode,
    SubprocessSpawnSpec,
)

_READ_CHUNK_BYTES = 65536
_FORCE_KILL_SIGNAL: int = getattr(signal_mod, "SIGKILL", signal_mod.SIGTERM)


def validate_subprocess_spec(spec: SubprocessSpawnSpec) -> None:
    """Validate graceMs, pre-aborted signal, and argv[0] before spawn."""
    grace = spec["graceMs"]
    if (
        isinstance(grace, bool)
        or not isinstance(grace, int | float)
        or not (0 < grace <= MAX_TIMER_DELAY_MS)
        or math.isnan(grace)
        or grace == float("inf")
    ):
        raise ValueError(
            f"subprocess graceMs must be a positive finite number no greater than {MAX_TIMER_DELAY_MS}"
        )
    sig = spec.get("signal")
    if sig is not None and sig.aborted:
        reason = "aborted" if sig.reason is None else str(sig.reason)
        raise RuntimeError(f"aborted before spawn: {reason}")
    argv = spec["argv"]
    if not argv or not argv[0]:
        raise ValueError("invalid argv: expected a non-empty program name at argv[0]")


def taskkill_process_tree(pid: int | None) -> None:
    """Force-terminate a Windows process tree via ``taskkill /T /F``."""
    if pid is None or pid <= 0:
        return
    try:
        py_subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=py_subprocess.DEVNULL,
            stderr=py_subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        pass


def signal_tree(pid: int | None, sig: int) -> None:
    """Send *sig* to the detached process group (or Windows tree via taskkill)."""
    if pid is None or pid <= 0:
        return
    if sys.platform == "win32":
        taskkill_process_tree(pid)
        return
    try:
        os.killpg(pid, sig)
    except OSError:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def _decode_returncode(returncode: int | None) -> SubprocessOutcome:
    if returncode is None:
        return {"exitCode": None, "signal": None}
    if returncode >= 0:
        return {"exitCode": returncode, "signal": None}
    sig_num = -returncode
    try:
        sig_name: str | None = signal_mod.Signals(sig_num).name
    except ValueError:
        sig_name = f"SIG{sig_num}"
    return {"exitCode": None, "signal": sig_name}


def _env_lookup_ci(env: Mapping[str, str], name: str) -> str | None:
    if name in env:
        return env[name]
    upper = name.upper()
    for key, val in env.items():
        if key.upper() == upper:
            return val
    return None


def _is_collect_mode(mode: SubprocessOutputMode) -> SubprocessCollect | None:
    if isinstance(mode, dict) and mode.get("kind") == "collect":
        return mode
    return None


def _linux_process_group_has_live_members(pgid: int) -> bool | None:
    try:
        entries = [p.name for p in Path("/proc").iterdir() if p.name.isdigit()]
    except OSError:
        return None
    matched = False
    for entry in entries:
        try:
            stat_text = Path(f"/proc/{entry}/stat").read_text(
                encoding="ascii", errors="replace"
            )
        except OSError:
            continue
        rparen = stat_text.rfind(")")
        if rparen == -1:
            continue
        fields = stat_text[rparen + 2 :].split()
        if len(fields) < 3:
            continue
        try:
            proc_pgrp = int(fields[2])
        except ValueError:
            continue
        if proc_pgrp != pgid:
            continue
        matched = True
        if fields[0] not in ("Z", "X", "x"):
            return True
    return False if matched else None


def _process_group_has_live_members(pid: int) -> bool:
    if sys.platform == "win32":
        return False
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    if sys.platform == "linux" and _linux_process_group_has_live_members(pid) is False:
        return False
    return True


def _write_stdin_sync(pipe: IO[bytes], data: bytes) -> None:
    try:
        pipe.write(data)
        pipe.flush()
    except OSError:
        pass
    finally:
        try:
            pipe.close()
        except OSError:
            pass


class _PipeStreamWriter:
    """Synchronous-on-creation StreamWriter adapter over ``Popen.stdin``."""

    def __init__(self, pipe: IO[bytes]) -> None:
        self._pipe = pipe
        self._closing = False

    def write(self, data: bytes) -> None:
        self._pipe.write(data)

    def writelines(self, lines: Sequence[bytes]) -> None:
        self._pipe.writelines(lines)

    def write_eof(self) -> None:
        self.close()

    def can_write_eof(self) -> bool:
        return True

    def is_closing(self) -> bool:
        return self._closing

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        try:
            self._pipe.flush()
        except OSError:
            pass
        try:
            self._pipe.close()
        except OSError:
            pass

    async def drain(self) -> None:
        await asyncio.to_thread(self._pipe.flush)

    async def wait_closed(self) -> None:
        return None


def _start_popen(
    argv: Sequence[str],
    kwargs: dict[str, Any],
) -> py_subprocess.Popen[bytes]:
    return py_subprocess.Popen(list(argv), **kwargs)


def _close_pipe(pipe: IO[bytes]) -> None:
    try:
        pipe.close()
    except OSError:
        pass


class _StreamPump:
    """Drain one ``Popen`` output pipe into an ``OutputCollector`` or ``StreamReader``."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        pipe: IO[bytes],
        on_data: Callable[[bytes], Any],
        on_eof: Callable[[], None],
    ) -> None:
        self._loop = loop
        self._pipe = pipe
        self._on_data = on_data
        self._on_eof = on_eof
        self.done = asyncio.Event()
        self._fd: int | None = None
        self._closed = False
        self._thread_task: asyncio.Task[None] | None = None

        if sys.platform != "win32":
            fd = pipe.fileno()
            os.set_blocking(fd, False)
            self._fd = fd
            loop.add_reader(fd, self._on_readable)
        else:
            self._thread_task = asyncio.create_task(self._read_threaded())

    def _finish(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._fd is not None:
            with contextlib.suppress(RuntimeError, OSError):
                self._loop.remove_reader(self._fd)
            self._fd = None
        _close_pipe(self._pipe)
        self._on_eof()
        self.done.set()

    def _on_readable(self) -> None:
        assert self._fd is not None
        try:
            chunk = os.read(self._fd, _READ_CHUNK_BYTES)
        except BlockingIOError:
            return
        except OSError:
            chunk = b""
        if not chunk:
            self._finish()
            return
        self._on_data(chunk)

    async def _read_threaded(self) -> None:
        try:
            while not self._closed:
                chunk = await asyncio.to_thread(self._pipe.read, _READ_CHUNK_BYTES)
                if not chunk:
                    break
                self._on_data(chunk)
        except OSError:
            pass
        finally:
            self._finish()

    def stop(self) -> None:
        """Stop reading (for collect grace-timeout cutoff) and release the pipe."""
        self._finish()


class LocalSubprocessHandle:
    """Managed async subprocess handle with bounded collectors and escalation."""

    def __init__(
        self,
        spec: SubprocessSpawnSpec,
        *,
        spill_dir: str,
        on_spill_failure: Callable[[BaseException, str], None],
        on_released: Callable[[LocalSubprocessHandle], None],
    ) -> None:
        self._spec = spec
        self._spill_dir = spill_dir
        self._on_spill_failure = on_spill_failure
        self._on_released = on_released
        self._loop = asyncio.get_running_loop()
        self._done: asyncio.Future[SubprocessOutcome] = self._loop.create_future()
        self._exited = asyncio.Event()
        self._popen: py_subprocess.Popen[bytes] | None = None
        self._stdin_writer: asyncio.StreamWriter | None = None
        self._stdout_reader: asyncio.StreamReader | None = None
        self._stderr_reader: asyncio.StreamReader | None = None
        self._stdin_task: asyncio.Task[None] | None = None
        self._collect_pumps: list[_StreamPump] = []
        self._all_pumps: list[_StreamPump] = []
        self._termination_started = False
        self._process_exited = False
        self._grace_handle: asyncio.TimerHandle | None = None
        self._unsub_abort: Callable[[], None] | None = None

        self._collected: SubprocessCollected = {}
        out_spec = _is_collect_mode(spec["stdio"]["stdout"])
        err_spec = _is_collect_mode(spec["stdio"]["stderr"])
        if out_spec is not None:
            self._collected["stdout"] = self._make_collector("stdout", out_spec)
        if err_spec is not None:
            self._collected["stderr"] = self._make_collector("stderr", err_spec)

        stdio = spec["stdio"]
        stdin_arg = (
            py_subprocess.DEVNULL if stdio["stdin"] == "ignore" else py_subprocess.PIPE
        )
        stdout_arg = None if stdio["stdout"] == "inherit" else py_subprocess.PIPE
        stderr_arg = None if stdio["stderr"] == "inherit" else py_subprocess.PIPE
        kwargs: dict[str, Any] = {
            "cwd": spec["cwd"],
            "env": child_env(spec.get("env")),
            "stdin": stdin_arg,
            "stdout": stdout_arg,
            "stderr": stderr_arg,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(
                py_subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200
            )
        else:
            kwargs["start_new_session"] = True

        try:
            popen = _start_popen(spec["argv"], kwargs)
        except OSError as exc:
            self._cleanup_after_exit()
            self._done.set_exception(exc)
            self._driver_task: asyncio.Task[None] | None = None
            return

        self._popen = popen
        if stdio["stdin"] == "pipe" and popen.stdin is not None:
            self._stdin_writer = cast(
                asyncio.StreamWriter, _PipeStreamWriter(popen.stdin)
            )
        elif isinstance(stdio["stdin"], dict) and popen.stdin is not None:
            data = stdio["stdin"]["data"]
            self._stdin_task = asyncio.create_task(
                asyncio.to_thread(_write_stdin_sync, popen.stdin, data)
            )

        out_coll = self._collected.get("stdout")
        if popen.stdout is not None:
            if isinstance(out_coll, OutputCollector):
                pump = _StreamPump(
                    self._loop, popen.stdout, out_coll.push, lambda: None
                )
                self._collect_pumps.append(pump)
                self._all_pumps.append(pump)
            else:
                reader = asyncio.StreamReader(loop=self._loop)
                self._stdout_reader = reader
                self._all_pumps.append(
                    _StreamPump(
                        self._loop, popen.stdout, reader.feed_data, reader.feed_eof
                    )
                )

        err_coll = self._collected.get("stderr")
        if popen.stderr is not None:
            if isinstance(err_coll, OutputCollector):
                pump = _StreamPump(
                    self._loop, popen.stderr, err_coll.push, lambda: None
                )
                self._collect_pumps.append(pump)
                self._all_pumps.append(pump)
            else:
                reader = asyncio.StreamReader(loop=self._loop)
                self._stderr_reader = reader
                self._all_pumps.append(
                    _StreamPump(
                        self._loop, popen.stderr, reader.feed_data, reader.feed_eof
                    )
                )

        sig = spec.get("signal")
        if sig is not None:
            self._unsub_abort = sig.on_abort(self.terminate)

        self._driver_task = asyncio.create_task(self._run())

    def _make_collector(self, label: str, cfg: SubprocessCollect) -> OutputCollector:
        spill_cfg = cfg.get("spill")
        spill = (
            SpillOptions(
                max_bytes=spill_cfg["maxBytes"],
                dir_path=self._spill_dir,
                on_failure=self._on_spill_failure,
            )
            if spill_cfg is not None
            else None
        )
        return OutputCollector(cfg["maxBytes"], label, spill)

    @property
    def pid(self) -> int | None:
        """Return child OS process ID if spawn succeeded."""
        return None if self._popen is None else self._popen.pid

    @property
    def stdin(self) -> asyncio.StreamWriter | None:
        """Exposed writer when ``stdio.stdin == 'pipe'``."""
        return self._stdin_writer

    @property
    def stdout(self) -> asyncio.StreamReader | None:
        """Exposed stdout StreamReader when configured as ``pipe``."""
        return self._stdout_reader

    @property
    def stderr(self) -> asyncio.StreamReader | None:
        """Exposed stderr StreamReader when configured as ``pipe``."""
        return self._stderr_reader

    @property
    def collected(self) -> SubprocessCollected:
        """Collected stream readers configured via ``collect``."""
        return self._collected

    @property
    def done(self) -> asyncio.Future[SubprocessOutcome]:
        """Future resolving with exit outcome or rejecting on spawn failure."""
        return self._done

    def _clear_grace(self) -> None:
        if self._grace_handle is not None:
            self._grace_handle.cancel()
            self._grace_handle = None

    def _force_kill(self) -> None:
        self._grace_handle = None
        if self._popen is None or not (
            not self._process_exited or _process_group_has_live_members(self._popen.pid)
        ):
            return
        signal_tree(self._popen.pid, _FORCE_KILL_SIGNAL)

    def terminate(self) -> None:
        """Start SIGTERM -> grace -> SIGKILL escalation for the spawned tree."""
        if self._termination_started or self._exited.is_set():
            return
        self._termination_started = True
        assert self._popen is not None
        signal_tree(self._popen.pid, signal_mod.SIGTERM)
        self._grace_handle = self._loop.call_later(
            self._spec["graceMs"] / 1000.0,
            self._force_kill,
        )

    def terminate_for_host_exit(self) -> None:
        """Synchronously force-terminate the managed process tree on host exit."""
        if self._exited.is_set() or self._popen is None:
            return
        signal_tree(self._popen.pid, _FORCE_KILL_SIGNAL)

    async def wait_for_exit(self, signal: AbortSignal | None = None) -> bool:
        """Wait until the managed process tree has exited, or return False if aborted."""
        if self._exited.is_set():
            return True
        if signal is None:
            await self._exited.wait()
            return True
        if signal.aborted:
            return False
        abort_fut: asyncio.Future[None] = self._loop.create_future()

        def _on_abort() -> None:
            abort_fut.set_result(None)

        unsub = signal.on_abort(_on_abort)
        wait_task = asyncio.create_task(self._exited.wait())
        try:
            await asyncio.wait(
                (wait_task, abort_fut),
                return_when=asyncio.FIRST_COMPLETED,
            )
            return not (abort_fut.done() and not wait_task.done())
        finally:
            unsub()
            if not wait_task.done():
                wait_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await wait_task

    async def _run(self) -> None:
        popen = self._popen
        assert popen is not None
        returncode = await asyncio.to_thread(popen.wait)
        self._process_exited = True

        if self._collect_pumps:
            pump_waits = [
                asyncio.create_task(p.done.wait()) for p in self._collect_pumps
            ]
            done_pumps, pending_pumps = await asyncio.wait(
                pump_waits,
                timeout=self._spec["graceMs"] / 1000.0,
            )
            del done_pumps
            for task in pending_pumps:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            for pump in self._collect_pumps:
                pump.stop()

        for key in ("stdout", "stderr"):
            coll = self._collected.get(key)
            if isinstance(coll, OutputCollector):
                coll.seal()

        self._done.set_result(_decode_returncode(returncode))

        while _process_group_has_live_members(popen.pid):
            await asyncio.sleep(0.01)

        if self._stdin_task is not None:
            with contextlib.suppress(BaseException):
                await self._stdin_task
        self._cleanup_after_exit()

    def _cleanup_after_exit(self) -> None:
        self._clear_grace()
        if self._unsub_abort is not None:
            self._unsub_abort()
            self._unsub_abort = None
        self._exited.set()
        self._on_released(self)


def _abort_error(signal: AbortSignal) -> BaseException:
    if isinstance(signal.reason, BaseException):
        return signal.reason
    msg = f"aborted: {signal.reason}" if signal.reason is not None else "aborted"
    return RuntimeError(msg)


class LocalSubprocessRuntime(SubprocessRuntime):
    """Host-OS subprocess runtime registering as ``ctx.subprocess``."""

    def __init__(
        self,
        ctx: Context,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(ctx)
        cfg = dict(config) if config is not None else {}
        spill_dir = cfg.get("spillDir")
        self._spill_dir = (
            spill_dir
            if isinstance(spill_dir, str) and spill_dir
            else default_spill_dir()
        )
        self._report_spill_failure = log_spill_failure(ctx.logger, "subprocess-local")
        self._live: set[LocalSubprocessHandle] = set()

        async def _dispose() -> None:
            targets = list(self._live)
            for handle in targets:
                handle.terminate()
            if targets:
                waits: list[Awaitable[Any]] = [h.wait_for_exit() for h in targets]
                await asyncio.gather(*waits, return_exceptions=True)
            self._live.clear()

        ctx.effect(lambda: _dispose, label="subprocess-local lifecycle")

    async def resolve_executable(
        self,
        command: str,
        env: Mapping[str, str] | None = None,
        signal: AbortSignal | None = None,
    ) -> str:
        """Resolve *command* to an executable path or fail loud."""
        if not command:
            raise ValueError("subprocess-local: executable must be non-empty")
        if signal is not None:
            signal.throw_if_aborted()
        environment = child_env(env)
        is_abs = os.path.isabs(command)
        if not is_abs and (
            "/" in command or (sys.platform == "win32" and "\\" in command)
        ):
            raise ValueError(
                f"subprocess-local: command {command!r} is a relative path; use an absolute path or a bare PATH name"
            )
        candidates = (
            [command] if is_abs else self._executable_candidates(command, environment)
        )
        for candidate in candidates:
            if signal is not None:
                signal.throw_if_aborted()
            p = Path(candidate)
            try:
                if not p.is_file():
                    continue
                if sys.platform != "win32" and not os.access(candidate, os.X_OK):
                    continue
                return candidate
            except OSError:
                continue
        if signal is not None:
            signal.throw_if_aborted()
        detail = (
            f"subprocess-local: command {command!r} is not an executable file"
            if is_abs
            else f"subprocess-local: command {command!r} was not found on PATH"
        )
        raise SubprocessExecutableNotFoundError(detail)

    def _executable_candidates(self, command: str, env: Mapping[str, str]) -> list[str]:
        path_val = _env_lookup_ci(env, "PATH") or ""
        ext = os.path.splitext(command)[1]
        if sys.platform == "win32" and ext == "":
            raw_pathext = _env_lookup_ci(env, "PATHEXT") or ".COM;.EXE;.BAT;.CMD"
            extensions = raw_pathext.split(";")
        else:
            extensions = [""]
        candidates: list[str] = []
        for entry in path_val.split(os.pathsep):
            if not entry:
                continue
            for suffix in extensions:
                candidates.append(os.path.join(entry, command + suffix))
        return candidates

    def spawn(self, spec: SubprocessSpawnSpec) -> LocalSubprocessHandle:
        """Validate *spec* and launch a tracked subprocess handle."""
        validate_subprocess_spec(spec)
        handle = LocalSubprocessHandle(
            spec,
            spill_dir=self._spill_dir,
            on_spill_failure=self._report_spill_failure,
            on_released=self._live.discard,
        )
        if not handle.done.done():
            self._live.add(handle)
        return handle


__all__ = [
    "LocalSubprocessHandle",
    "LocalSubprocessRuntime",
    "signal_tree",
    "taskkill_process_tree",
    "validate_subprocess_spec",
]
