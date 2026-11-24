"""Abstract FileSystem Service Definition (ctx.fs)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any

from animas import Context, Service

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.fs.patch import PatchHunk
from magitech_harness.fs.types import (
    FsDeleteOutcome,
    FsDirEntry,
    FsEditOutcome,
    FsEditRequest,
    FsError,
    FsInfo,
    FsMoveOutcome,
    FsPathInfo,
    FsTarget,
    FsVersion,
    FsWriteIntent,
    FsWriteOutcome,
)


class FileSystem(Service, ABC):
    """Abstract filesystem service registering as ``ctx.fs``."""

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx, "fs")

    @property
    def sandbox_mode(self) -> str | None:
        """Default sandbox confinement mode, or None when unconfined."""
        return None

    @property
    def sandboxMode(self) -> str | None:
        """CamelCase alias for :attr:`sandbox_mode`."""
        return self.sandbox_mode

    async def watch(
        self,
        target: FsTarget,
        changed: Callable[[Exception | None], None],
        signal: AbortSignal,
    ) -> Callable[[], Awaitable[None]]:
        """Observe changes to *target*; default raises FS_IO_ERROR."""
        del target, changed
        if signal.aborted:
            raise FsError("watch aborted", "FS_ABORTED")
        raise FsError(
            "Filesystem watching is not supported by this provider.",
            "FS_IO_ERROR",
        )

    @abstractmethod
    async def resolve(
        self, path: str, opts: Mapping[str, Any] | None = None
    ) -> FsTarget:
        """Resolve *path* into a stable target identity."""
        ...

    @abstractmethod
    def process_path(self, target: FsTarget) -> str:
        """Return canonical absolute path usable by child processes."""
        ...

    def processPath(self, target: FsTarget) -> str:
        """CamelCase alias for :meth:`process_path`."""
        return self.process_path(target)

    def process_path_from_host_path(self, host_path: str) -> str | None:
        """Map an absolute host path into this execution world, if shared."""
        del host_path
        return None

    def processPathFromHostPath(self, host_path: str) -> str | None:
        """CamelCase alias for :meth:`process_path_from_host_path`."""
        return self.process_path_from_host_path(host_path)

    @abstractmethod
    def file_url(self, target: FsTarget) -> str:
        """Return canonical ``file:`` URI for *target*."""
        ...

    def fileUrl(self, target: FsTarget) -> str:
        """CamelCase alias for :meth:`file_url`."""
        return self.file_url(target)

    @abstractmethod
    def contains(self, parent: FsTarget, child: FsTarget) -> bool:
        """Return True when *child* is *parent* or a descendant of it."""
        ...

    @abstractmethod
    async def stat(
        self, target: FsTarget, signal: AbortSignal | None = None
    ) -> FsInfo | None:
        """Return target metadata following symlinks, or None when absent."""
        ...

    @abstractmethod
    async def lstat(
        self,
        path: str,
        opts: Mapping[str, Any] | None = None,
        signal: AbortSignal | None = None,
    ) -> FsPathInfo | None:
        """Return path metadata without following a trailing symlink."""
        ...

    @abstractmethod
    async def read_text(
        self, target: FsTarget, signal: AbortSignal | None = None
    ) -> str:
        """Read and validate the whole file as UTF-8 text."""
        ...

    async def readText(
        self, target: FsTarget, signal: AbortSignal | None = None
    ) -> str:
        """CamelCase alias for :meth:`read_text`."""
        return await self.read_text(target, signal)

    @abstractmethod
    async def stream_text(
        self, target: FsTarget, signal: AbortSignal | None = None
    ) -> AsyncIterator[str]:
        """Stream decoded UTF-8 text chunks for *target*."""
        ...

    async def streamText(
        self, target: FsTarget, signal: AbortSignal | None = None
    ) -> AsyncIterator[str]:
        """CamelCase alias for :meth:`stream_text`."""
        return await self.stream_text(target, signal)

    @abstractmethod
    async def read_bytes(
        self,
        target: FsTarget,
        signal: AbortSignal | None,
        max_bytes: int,
    ) -> bytes:
        """Read raw bytes up to *max_bytes*, failing FS_TOO_LARGE if exceeded."""
        ...

    async def readBytes(
        self,
        target: FsTarget,
        signal: AbortSignal | None,
        max_bytes: int,
    ) -> bytes:
        """CamelCase alias for :meth:`read_bytes`."""
        return await self.read_bytes(target, signal, max_bytes)

    @abstractmethod
    async def read_byte_range(
        self,
        target: FsTarget,
        byte_range: Mapping[str, int],
        signal: AbortSignal | None = None,
    ) -> bytes:
        """Read raw byte slice ``[offset, offset + length)``."""
        ...

    async def readByteRange(
        self,
        target: FsTarget,
        byte_range: Mapping[str, int],
        signal: AbortSignal | None = None,
    ) -> bytes:
        """CamelCase alias for :meth:`read_byte_range`."""
        return await self.read_byte_range(target, byte_range, signal)

    @abstractmethod
    async def list_dir(
        self, target: FsTarget, signal: AbortSignal | None = None
    ) -> list[FsDirEntry]:
        """List direct directory children in deterministic name order."""
        ...

    async def listDir(
        self, target: FsTarget, signal: AbortSignal | None = None
    ) -> list[FsDirEntry]:
        """CamelCase alias for :meth:`list_dir`."""
        return await self.list_dir(target, signal)

    @abstractmethod
    async def write_text(
        self,
        target: FsTarget,
        content: str,
        expected: FsWriteIntent | None = None,
        signal: AbortSignal | None = None,
        sandbox_policy: Any = None,
    ) -> FsWriteOutcome:
        """Atomically create or replace UTF-8 text."""
        ...

    async def writeText(
        self,
        target: FsTarget,
        content: str,
        expected: FsWriteIntent | None = None,
        signal: AbortSignal | None = None,
        sandbox_policy: Any = None,
    ) -> FsWriteOutcome:
        """CamelCase alias for :meth:`write_text`."""
        return await self.write_text(target, content, expected, signal, sandbox_policy)

    @abstractmethod
    async def edit_text(
        self,
        target: FsTarget,
        edit: FsEditRequest,
        expected: Mapping[str, FsVersion] | None = None,
        signal: AbortSignal | None = None,
        sandbox_policy: Any = None,
    ) -> FsEditOutcome:
        """Atomically apply a literal search/replace edit."""
        ...

    async def editText(
        self,
        target: FsTarget,
        edit: FsEditRequest,
        expected: Mapping[str, FsVersion] | None = None,
        signal: AbortSignal | None = None,
        sandbox_policy: Any = None,
    ) -> FsEditOutcome:
        """CamelCase alias for :meth:`edit_text`."""
        return await self.edit_text(target, edit, expected, signal, sandbox_policy)

    @abstractmethod
    async def patch_text(
        self,
        target: FsTarget,
        hunks: Sequence[PatchHunk],
        expected: Mapping[str, FsVersion] | None = None,
        signal: AbortSignal | None = None,
        sandbox_policy: Any = None,
    ) -> FsEditOutcome:
        """Atomically apply parsed hunks to *target*."""
        ...

    async def patchText(
        self,
        target: FsTarget,
        hunks: Sequence[PatchHunk],
        expected: Mapping[str, FsVersion] | None = None,
        signal: AbortSignal | None = None,
        sandbox_policy: Any = None,
    ) -> FsEditOutcome:
        """CamelCase alias for :meth:`patch_text`."""
        return await self.patch_text(target, hunks, expected, signal, sandbox_policy)

    @abstractmethod
    async def delete_file(
        self,
        target: FsTarget,
        expected: Mapping[str, FsVersion] | None = None,
        signal: AbortSignal | None = None,
        sandbox_policy: Any = None,
    ) -> FsDeleteOutcome:
        """Remove a regular file with optional version guard."""
        ...

    async def deleteFile(
        self,
        target: FsTarget,
        expected: Mapping[str, FsVersion] | None = None,
        signal: AbortSignal | None = None,
        sandbox_policy: Any = None,
    ) -> FsDeleteOutcome:
        """CamelCase alias for :meth:`delete_file`."""
        return await self.delete_file(target, expected, signal, sandbox_policy)

    @abstractmethod
    async def move_file(
        self,
        source: FsTarget,
        destination: FsTarget,
        options: Mapping[str, Any] | None = None,
        signal: AbortSignal | None = None,
        sandbox_policy: Any = None,
    ) -> FsMoveOutcome:
        """Atomically publish *destination* from *source*, then remove *source*."""
        ...

    async def moveFile(
        self,
        source: FsTarget,
        destination: FsTarget,
        options: Mapping[str, Any] | None = None,
        signal: AbortSignal | None = None,
        sandbox_policy: Any = None,
    ) -> FsMoveOutcome:
        """CamelCase alias for :meth:`move_file`."""
        return await self.move_file(
            source, destination, options, signal, sandbox_policy
        )


__all__ = ["FileSystem"]
