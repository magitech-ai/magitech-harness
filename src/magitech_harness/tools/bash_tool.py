"""Base shell tool: ``bash``.

Ports dsh's ``@deepseek-ai/dsh-tool-bash`` at
``packages/shell/tool-bash/src/index.ts``. Commands run through ``ctx.shell``
when the service is mounted (bounded in-memory output, process-tree termination,
cooperative abort) and through a local subprocess otherwise. A truncated stream
is never silently dropped: the complete output stays on disk and the rendered
result names the spill file.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from animas import Context

from magitech_harness.sandbox.escalation import (
    escalation_hint_marker,
    sandbox_denial_marker,
)
from magitech_harness.tools.schema import define_tool

DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024  # 64 KiB in-memory limit
DEFAULT_TIMEOUT_MS = 30_000

BASH_SECTION_TEXT = (
    "Use the bash tool to execute shell commands. Commands run in a "
    "subprocess with a configurable timeout; large output is spilled to disk. "
    "Prefer read_file/write_file/edit_file for file access over shell commands."
)


async def handle_bash(
    args: dict[str, Any],
    ctx: Context | None = None,
    spill_dir: Path | None = None,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    exec_view: Any = None,
) -> str:
    """Execute a shell command with spill-to-disk recovery.

    ``args`` carries ``command`` and optional ``timeout_ms`` and ``cwd``.
    """
    command = args.get("command")
    if not command or not command.strip():
        return "Error: 'command' argument is required."

    timeout_ms = args.get("timeout_ms", DEFAULT_TIMEOUT_MS)
    timeout_seconds = max(0.001, timeout_ms / 1000)

    cwd_str = args.get("cwd")
    cwd = Path(cwd_str) if cwd_str else None
    if cwd and not cwd.is_dir():
        return f"Error: Working directory does not exist: {cwd_str}"

    sandbox_policy: dict[str, Any] | None = None
    denial_note = ""
    if ctx is not None and ctx.has("sandboxPolicy"):
        from magitech_harness.sandbox.escalation import (
            ESCALATION_TARGETS,
            approve_escalation,
            validate_escalation_args,
        )

        try:
            validate_escalation_args(
                args.get("sandbox_permissions"), args.get("justification")
            )
        except Exception as error:  # noqa: BLE001 -- invalid escalation args are user errors
            return f"Error: {error}"
        agent = exec_view.get("agent") if isinstance(exec_view, dict) else None
        session = getattr(agent, "session", None)
        standing = ctx.sandboxPolicy.resolve(session)
        requested = args.get("sandbox_permissions")
        policy = dict(standing)
        if requested is not None:
            if requested not in ESCALATION_TARGETS:
                return f"Error: invalid sandbox_permissions {requested!r}"
            try:
                mode = await approve_escalation(
                    {
                        "requestedMode": requested,
                        "effectiveMode": standing["mode"],
                        "justification": args.get("justification"),
                        "subject": "command",
                    },
                    {
                        "approver": ctx.approval if ctx.has("approval") else None,
                        "agent": agent,
                        "callId": (exec_view or {}).get("callId"),
                        "toolName": "bash",
                        "signal": (exec_view or {}).get("signal"),
                    },
                )
            except Exception as error:  # noqa: BLE001 -- escalation failure is surfaced to the model
                return f"Error: {error}"
            policy["mode"] = mode
            policy["approved"] = True
        sandbox_policy = policy

    truncated = False
    spill_paths: list[str] = []
    if ctx is not None and ctx.has("shell"):
        run_kwargs: dict[str, Any] = {
            "cwd": cwd,
            "timeout_seconds": timeout_seconds,
            "max_output_bytes": max_output_bytes,
        }
        if sandbox_policy is not None:
            run_kwargs["sandboxPolicy"] = sandbox_policy
        try:
            res = await ctx.shell.run(command, **run_kwargs)
        except TimeoutError:
            return f"Command timed out after {timeout_seconds} seconds."
        exit_code = res.exit_code
        stdout = res.stdout.strip()
        stderr = res.stderr.strip()
        timed_out = res.timed_out
        # The shell collector already spilled the COMPLETE stream to disk;
        # carry its truncation and spill paths into the rendered marker.
        truncated = (
            bool(getattr(res, "truncated", False))
            or "[truncated" in stdout
            or "[truncated" in stderr
        )
        spill_paths = [
            p
            for p in (
                getattr(res, "stdout_spill_path", None),
                getattr(res, "stderr_spill_path", None),
            )
            if p
        ]
    else:
        return "Error: no shell executor is mounted on ctx.shell"

    if timed_out:
        return f"Command timed out after {timeout_seconds} seconds."

    spill_note = ""
    if truncated or spill_paths:
        if not spill_paths:  # pragma: no cover - collector always spills
            # No collector spill (e.g. no ctx.shell path): write the output we
            # have. The collector path never reaches here with truncated=True
            # and no spill, since it always spills the complete stream.
            effective_spill_dir = spill_dir or (Path.home() / ".agents" / "spill")
            try:
                effective_spill_dir.mkdir(parents=True, exist_ok=True)
                spill_file = effective_spill_dir / f"{uuid.uuid4()}.stdout"
                spill_file.write_text(
                    stdout + (f"\n{stderr}" if stderr else ""), encoding="utf-8"
                )
                spill_paths = [str(spill_file)]
            except OSError:
                pass
        # Exact reference marker (tool-bash/src/render.ts): never silently drop.
        full = ", ".join(spill_paths) if spill_paths else "(unavailable)"
        spill_note = f"\n[output truncated; full output: {full}]"

    facts = (
        getattr(res, "sandbox", None) if ctx is not None and ctx.has("shell") else None
    )
    if isinstance(facts, dict) and facts.get("denied"):
        mode = str(facts.get("mode", "read-only"))
        denial_note = (
            f"\n{sandbox_denial_marker(mode)}\n{escalation_hint_marker('command')}"
        )

    if exit_code == 0:
        base = stdout if stdout else "(Command executed with no output)"
        return base + spill_note + denial_note

    output_body = stderr or stdout or "No output produced"
    return (
        f"Command failed with exit code {exit_code}:\n{output_body}{spill_note}"
        + denial_note
    )


def apply(ctx: Context, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
    """Mount the ``bash`` tool onto ``ctx.tools`` (dsh's ``tool-bash`` row)."""
    del config, kwargs

    async def _execute(args: dict[str, Any], exec_view: Any = None) -> str:
        return await handle_bash(args, ctx=ctx, exec_view=exec_view)

    ctx.tools.register(
        define_tool(
            name="bash",
            description="Execute a bash command in a subprocess.",
            parameters={
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                    "required": True,
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": (
                        f"Timeout in milliseconds (default {DEFAULT_TIMEOUT_MS})"
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory",
                },
                "sandbox_permissions": {
                    "type": "string",
                    "enum": ["workspace-write", "danger-full-access"],
                    "description": "The narrowest wider sandbox mode for a one-shot retry of the exact command the sandbox just denied; the retry asks the user for approval.",
                },
                "justification": {
                    "type": "string",
                    "description": "Required with sandbox_permissions: one sentence for the user explaining why this exact command needs the wider access.",
                },
            },
            execute=_execute,
        )
    )

    system_prompt = ctx.get("systemPrompt", None, strict=False)
    if system_prompt is not None:
        system_prompt.section(
            {
                "name": "tool:bash",
                "order": system_prompt.get_section_order("TOOL_BASH"),
                "text": BASH_SECTION_TEXT,
            }
        )


apply.inject = ["tools"]  # type: ignore[attr-defined]

__all__ = [
    "BASH_SECTION_TEXT",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_TIMEOUT_MS",
    "apply",
    "handle_bash",
]
