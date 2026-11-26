"""Model-facing PowerShell tool (``pwsh``).

Ports dsh's ``@deepseek-ai/dsh-tool-pwsh`` at ``packages/shell/tool-pwsh/src``.
The PowerShell dialect is native: ``pwsh -Command``, ``$env:NAME`` variables,
and Windows-style path examples. Commands run in a fresh ``pwsh -Command``
subprocess with bounded output, spill recovery and cooperative cancellation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from animas import Context

from magitech_harness.shell._subprocess_base import (
    DEFAULT_MAX_TIMEOUT_MS,
    assert_positive_finite,
)
from magitech_harness.tools.schema import define_tool

DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024
DEFAULT_TIMEOUT_MS = 30_000

PWSH_DESCRIPTION = (
    "Execute a PowerShell command (`pwsh -Command`) and return its stdout/stderr. "
    "Each call runs in a fresh pwsh process; pass `workdir` instead of using `cd`. "
    "Paths use native Windows form (`C:\\...`); read environment variables with "
    "`$env:NAME`. Managed `$env:DSH_*` variables expose current harness environment facts. "
    "Long output is truncated to its tail; the full output is saved to a file whose path "
    "is reported when available. On Windows a force-killed command settles as "
    "`[exit code: 1]` without a signal marker — treat it as an interruption, not a command failure. "
    "Provide `description` before `command` in the arguments. "
    "Before any delete or move, verify that the resolved absolute target path is the "
    "intended one; never run it against a computed path you have not checked. "
    "Do not assign to automatic variables such as `$HOME`; variable names are "
    "case-insensitive, so `$home` is the same read-only variable. "
    "Commands may run under a file sandbox; a blocked file operation is reported as "
    "`[sandbox: file access denied under <mode> mode]`, a policy denial: do not retry another way."
)

PWSH_SECTION = (
    "Non-zero exits are reported as `[exit code: N]` markers; investigate failures before "
    "moving on. On Windows a killed process settles as `[exit code: 1]` without a signal "
    "marker; treat a bare exit 1 after an interruption as a termination, not a command failure."
)


async def handle_pwsh(
    args: dict[str, Any],
    ctx: Context | None = None,
    spill_dir: Path | None = None,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    timeout_cap_ms: float = DEFAULT_MAX_TIMEOUT_MS,
    exec_view: Any = None,
) -> str:
    """Execute one PowerShell command and render its result."""
    import asyncio
    import math
    import signal as signal_mod
    import uuid

    from magitech_harness.shell.pwsh_local import (
        ENCODING_PREAMBLE,
        resolve_pwsh_path,
    )

    command = args.get("command")
    if not command or not str(command).strip():
        return "Error: 'command' argument is required."

    timeout_ms = args.get("timeoutMs", args.get("timeout_ms", DEFAULT_TIMEOUT_MS))
    if (
        isinstance(timeout_ms, bool)
        or not isinstance(timeout_ms, (int, float))
        or not math.isfinite(timeout_ms)
        or timeout_ms <= 0
    ):
        return "Error: 'timeoutMs' must be a finite positive number."
    timeout_seconds = min(float(timeout_ms), timeout_cap_ms) / 1000

    workdir = args.get("workdir", args.get("cwd"))
    cwd = Path(workdir) if workdir else None
    if cwd is not None and not cwd.is_dir():
        return f"Error: Working directory does not exist: {workdir}"

    configured = None
    if ctx is not None:
        for key in ("pwshConfig", "pwshPath"):
            if ctx.has(key):
                configured = ctx.get(key, None, strict=False)
                break
    executable = resolve_pwsh_path(configured)
    argv = [
        executable,
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        ENCODING_PREAMBLE + str(command),
    ]
    signal = (
        exec_view.get("signal")
        if isinstance(exec_view, dict) and exec_view.get("signal") is not None
        else args.get("signal")
    )
    denial_signatures: tuple[str, ...] = ()
    if ctx is not None and ctx.has("sandbox") and ctx.has("sandboxPolicy"):
        agent = exec_view.get("agent") if isinstance(exec_view, dict) else None
        session = getattr(agent, "session", None)
        policy = ctx.sandboxPolicy.resolve(session)
        if policy.get("mode") != "danger-full-access":
            confined = ctx.sandbox.confine(argv, policy, signal)
            argv = confined.argv
            denial_signatures = tuple(getattr(confined, "denial_signatures", ()))
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        communicate = asyncio.ensure_future(process.communicate())
        wait_for: set[Any] = {communicate}
        if signal is not None and hasattr(signal, "add_listener"):
            aborted = asyncio.get_running_loop().create_future()

            def on_abort() -> None:
                if not aborted.done():
                    aborted.set_result(None)

            dispose = signal.add_listener(on_abort)
            wait_for.add(aborted)
        else:
            dispose = None
        try:
            done, _pending = await asyncio.wait(
                wait_for,
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if "aborted" in locals() and aborted in done:
                communicate.cancel()
                try:
                    process.kill()
                    await process.wait()
                except OSError:
                    pass
                return "Command aborted."
            if communicate not in done:
                communicate.cancel()
                try:
                    process.kill()
                    await process.wait()
                except OSError:
                    pass
                return f"[timed out after {int(timeout_seconds * 1000)}ms]"
            stdout_bytes, stderr_bytes = communicate.result()
        finally:
            if dispose is not None:
                dispose()
    except OSError as error:
        return f"Error: could not start PowerShell: {error}"

    exit_code = process.returncode if process.returncode is not None else -1
    # A negative POSIX return code means the subprocess itself was signalled;
    # this is independent of the caller's cooperative AbortSignal.
    process_signal: str | None = None
    if exit_code < 0:
        try:
            process_signal = signal_mod.Signals(-exit_code).name
        except ValueError:
            process_signal = str(-exit_code)
    spills: dict[str, str] = {}
    stdout_truncated = len(stdout_bytes) > max_output_bytes
    stderr_truncated = len(stderr_bytes) > max_output_bytes
    if stdout_truncated or stderr_truncated:
        target_dir = spill_dir or (Path.home() / ".magitech" / "spill")
        for name, data, exceeded in (
            ("stdout", stdout_bytes, stdout_truncated),
            ("stderr", stderr_bytes, stderr_truncated),
        ):
            if not exceeded:
                continue
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                path = target_dir / f"{uuid.uuid4()}.{name}"
                path.write_bytes(data)
                spills[name] = str(path)
            except OSError:
                pass

    def stream_text(name: str, data: bytes, truncated: bool) -> str:
        text = data[-max_output_bytes:].decode("utf-8", errors="replace").strip()
        if truncated:
            full = spills.get(name, "(unavailable)")
            text = f"{text}\n[output truncated; full output: {full}]"
        return text

    body = stream_text("stdout", stdout_bytes, stdout_truncated)
    error_text = stream_text("stderr", stderr_bytes, stderr_truncated)
    if error_text:
        body = f"{body}\n[stderr]\n{error_text}" if body else f"[stderr]\n{error_text}"
    if not body:
        body = "(no output)"

    markers: list[str] = []
    if denial_signatures and exit_code not in (0, None):
        lowered = error_text.lower()
        if any(
            signature.lower() in lowered
            for signature in denial_signatures
            if signature.strip()
        ):
            from magitech_harness.sandbox.escalation import (
                escalation_hint_marker,
                sandbox_denial_marker,
            )

            markers.append(sandbox_denial_marker(str(policy.get("mode", "read-only"))))
            markers.append(escalation_hint_marker("command"))
    # The exit markers stay last: status parsers anchor there.
    if process_signal is not None:
        markers.append(f"[killed by signal: {process_signal}]")
    elif exit_code != 0:
        markers.append(f"[exit code: {exit_code}]")
    if markers:
        body += "\n" + "\n".join(markers)
    return body


def apply(ctx: Context, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
    """Mount the ``pwsh`` tool onto ``ctx.tools``."""
    del config, kwargs

    async def _execute(args: dict[str, Any], exec_view: Any = None) -> str:
        shell = ctx.get("shell", None, strict=False)
        shell_config = getattr(shell, "config", None) or {}
        cap = assert_positive_finite(
            "pwsh",
            "maxTimeoutMs",
            shell_config.get("maxTimeoutMs", DEFAULT_MAX_TIMEOUT_MS),
        )
        return await handle_pwsh(args, ctx=ctx, exec_view=exec_view, timeout_cap_ms=cap)

    ctx.tools.register(
        define_tool(
            name="pwsh",
            description=PWSH_DESCRIPTION,
            parameters={
                "description": {
                    "type": "string",
                    "required": True,
                    "description": "Clear, concise description of what this command does in active voice.",
                },
                "command": {
                    "type": "string",
                    "required": True,
                    "description": "The PowerShell command to execute.",
                },
                "timeoutMs": {
                    "type": "number",
                    "description": (
                        "Timeout in milliseconds (bounded by maxTimeoutMs); the "
                        "command is killed on expiry."
                    ),
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory; relative paths resolve against the session workspace.",
                },
            },
            execute=_execute,
        )
    )

    system_prompt = ctx.get("systemPrompt", None, strict=False)
    if system_prompt is not None:
        system_prompt.section(
            {
                "name": "tool:pwsh",
                "order": system_prompt.get_section_order("TOOL_PWSH"),
                "text": PWSH_SECTION,
                "retirement": "The pwsh tool is no longer available. Its guidance no longer applies.",
            }
        )


apply.inject = ["tools"]  # type: ignore[attr-defined]

__all__ = [
    "PWSH_DESCRIPTION",
    "PWSH_SECTION",
    "apply",
    "handle_pwsh",
]
