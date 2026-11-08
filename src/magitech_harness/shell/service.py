"""
LocalShellService (ctx.shell): Process and command execution provider.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from anima import Context, Service


class LocalShellService(Service):
    """Service providing shell command execution backend."""

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx, "shell")

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
