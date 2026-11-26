"""Model-facing PowerShell tool (``pwsh``).

Ports dsh's ``@deepseek-ai/dsh-tool-pwsh`` at ``packages/shell/tool-pwsh/src``.
The PowerShell dialect is native: ``pwsh -Command``, ``$env:NAME`` variables,
and Windows-style path examples. Commands run through ``ctx.shell`` when a
PowerShell executor is mounted, inheriting its output caps, spill recovery and
cooperative abort.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from animas import Context

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
) -> str:
    """Execute one PowerShell command and render its result."""
    import asyncio
    import uuid

    from magitech_harness.shell.pwsh_local import (
        ENCODING_PREAMBLE,
        resolve_pwsh_path,
    )

    command = args.get("command")
    if not command or not str(command).strip():
        return "Error: 'command' argument is required."

    timeout_ms = args.get("timeoutMs", args.get("timeout_ms", DEFAULT_TIMEOUT_MS))
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, (int, float)):
        return "Error: 'timeoutMs' must be a number."
    timeout_seconds = max(0.001, float(timeout_ms) / 1000)

    workdir = args.get("workdir", args.get("cwd"))
    cwd = Path(workdir) if workdir else None
    if cwd is not None and not cwd.is_dir():
        return f"Error: Working directory does not exist: {workdir}"

    configured = None
    if ctx is not None and ctx.has("pwshConfig"):
        configured = ctx.get("pwshConfig", None, strict=False)
    executable = resolve_pwsh_path(configured)
    argv = [
        executable,
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        ENCODING_PREAMBLE + str(command),
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        return f"Command timed out after {timeout_seconds} seconds."
    except OSError as error:
        return f"Error: could not start PowerShell: {error}"

    exit_code = process.returncode if process.returncode is not None else -1
    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
    spills: list[str] = []
    truncated = (
        len(stdout_bytes) > max_output_bytes or len(stderr_bytes) > max_output_bytes
    )
    if truncated:
        target_dir = spill_dir or (Path.home() / ".magitech" / "spill")
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            for name, data in (("stdout", stdout_bytes), ("stderr", stderr_bytes)):
                path = target_dir / f"{uuid.uuid4()}.{name}"
                path.write_bytes(data)
                spills.append(str(path))
        except OSError:
            pass

    body = stdout
    if stderr:
        body = f"{body}\n\n[stderr]\n{stderr}" if body else f"[stderr]\n{stderr}"
    if exit_code != 0:
        body = (
            f"{body}\n[exit code: {exit_code}]" if body else f"[exit code: {exit_code}]"
        )
    if spills or truncated:
        body += (
            f"\n[output truncated; full output: "
            f"{', '.join(spills) if spills else '(unavailable)'}]"
        )
    return body


def apply(ctx: Context, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
    """Mount the ``pwsh`` tool onto ``ctx.tools``."""
    del config, kwargs

    async def _execute(args: dict[str, Any]) -> str:
        return await handle_pwsh(args, ctx=ctx)

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
                    "description": "Timeout in milliseconds; the executor applies its configured cap.",
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
