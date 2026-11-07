"""
LocalShellService (ctx.shell): Process and command execution provider.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from kernel import Context, Service
from harness.tools.definition import define_tool


class LocalShellService(Service):
    """Service providing shell command execution and bash tool registration."""

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx, "shell")
        # Automatically register a bash tool if ctx.tools is loaded or when it becomes available
        self._register_shell_tool()

    def _register_shell_tool(self) -> None:
        if self.ctx.has("tools"):
            self.ctx.tools.register(
                define_tool(
                    name="bash",
                    description="Execute a shell command locally in bash and return its output.",
                    parameters={
                        "command": {
                            "type": "string",
                            "description": "The exact shell command line string to execute.",
                            "required": True,
                        }
                    },
                    execute=self._execute_bash_tool,
                )
            )

    async def _execute_bash_tool(self, args: dict) -> str:
        cmd = args.get("command", "")
        code, stdout, stderr = await self.run(cmd)
        output = []
        if stdout:
            output.append(stdout)
        if stderr:
            output.append(f"STDERR:\n{stderr}")
        output.append(f"[Exit code: {code}]")
        return "\n".join(output)

    async def run(
        self, command: str, cwd: Path | str | None = None
    ) -> tuple[int, str, str]:
        """Execute a shell command asynchronously."""
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        return (
            proc.returncode if proc.returncode is not None else -1,
            stdout_bytes.decode(errors="replace"),
            stderr_bytes.decode(errors="replace"),
        )
