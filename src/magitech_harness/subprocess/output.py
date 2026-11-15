"""Bounded tail output collector with best-effort atomic spill files."""

from __future__ import annotations

import os
import secrets
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from magitech_harness.subprocess.types import CollectedOutput, OutputReadSlice

_spill_counter = 0


def default_spill_dir() -> str:
    """Return private per-host-PID spill directory path under the OS temp dir."""
    return os.path.join(tempfile.gettempdir(), f"magitech-subprocess-{os.getpid()}")


@dataclass(slots=True)
class SpillOptions:
    """Configuration enabling spill-file capture alongside tail collection."""

    max_bytes: int
    dir_path: str
    on_failure: Callable[[BaseException, str], None]


def log_spill_failure(
    logger: Any, component: str = "subprocess-local"
) -> Callable[[BaseException, str], None]:
    """Create a spill failure reporter logging warnings via *logger*."""

    def _report(error: BaseException, label: str) -> None:
        logger.warning(
            f"{component}: failed to write {label} spill file; continuing with bounded tail: {error}"
        )

    return _report


class OutputCollector:
    """Collect a single byte stream keeping a byte-exact in-memory tail and optional spill."""

    def __init__(
        self,
        max_bytes: int,
        label: str,
        spill: SpillOptions | None = None,
    ) -> None:
        self._max_bytes = max(0, max_bytes)
        self._label = label
        self._spill = spill
        self._spill_disabled = spill is None
        self._spill_fd: int | None = None
        self._spill_file: str | None = None
        self._chunks: list[bytes] = []
        self._bytes = 0
        self._dropped = False
        self._total = 0

    @property
    def total_bytes(self) -> int:
        """Total bytes ingested so far (including dropped prefixes)."""
        return self._total

    def push(self, chunk: bytes) -> None:
        """Ingest one raw chunk from a stream, spilling if configured and overflowing."""
        self._total += len(chunk)
        overflows = (self._bytes + len(chunk)) > self._max_bytes
        if (
            self._spill is not None
            and not self._spill_disabled
            and (overflows or self._spill_fd is not None)
        ):
            self._spill_all(self._spill, chunk)
        if not chunk:
            return
        self._chunks.append(chunk)
        self._bytes += len(chunk)
        while self._bytes > self._max_bytes:
            head = self._chunks[0]
            excess = self._bytes - self._max_bytes
            if len(head) <= excess:
                self._chunks.pop(0)
                self._bytes -= len(head)
            else:
                self._chunks[0] = head[excess:]
                self._bytes -= excess
            self._dropped = True

    def _spill_all(self, spill: SpillOptions, chunk: bytes) -> None:
        global _spill_counter
        if self._total > spill.max_bytes:
            self._discard_spill()
            return
        try:
            if self._spill_fd is None:
                os.makedirs(spill.dir_path, mode=0o700, exist_ok=True)
                _spill_counter += 1
                filename = (
                    f"magitech-subprocess-{os.getpid()}-{_spill_counter}-"
                    f"{secrets.token_hex(6)}-{self._label}.log"
                )
                file_path = os.path.join(spill.dir_path, filename)
                fd = os.open(
                    file_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                self._spill_file = file_path
                self._spill_fd = fd
                for prior in self._chunks:
                    os.write(fd, prior)
            if chunk:
                os.write(self._spill_fd, chunk)
        except BaseException as exc:  # noqa: BLE001 - best-effort spill in stream callback
            self._discard_spill()
            spill.on_failure(exc, self._label)

    def _discard_spill(self) -> None:
        fd = self._spill_fd
        file_path = self._spill_file
        self._spill_fd = None
        self._spill_file = None
        self._spill_disabled = True
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                self._spill_fd = fd
        if file_path is not None:
            try:
                os.unlink(file_path)
            except OSError:
                pass

    def read_from(self, from_byte: int) -> OutputReadSlice:
        """Incremental UTF-8 read in whole-stream byte coordinates."""
        window_start = self._total - self._bytes
        buf = b"".join(self._chunks)
        lossy = from_byte < window_start
        chunk = buf if lossy else buf[from_byte - window_start :]
        result: OutputReadSlice = {
            "text": chunk.decode("utf-8", errors="replace"),
            "nextOffset": self._total,
            "lossy": lossy,
        }
        if self._spill_file is not None:
            result["spillPath"] = self._spill_file
        return result

    def snapshot(self) -> tuple[bytes, int]:
        """Return retained tail bytes and total observed stream byte count."""
        return b"".join(self._chunks), self._total

    def seal(self) -> None:
        """Close the active spill file descriptor once the stream ends."""
        if self._spill_fd is None:
            return
        try:
            os.close(self._spill_fd)
        except OSError:
            self._spill_file = None
        self._spill_fd = None

    def finalize(self) -> CollectedOutput:
        """Seal spill storage and return final tail text + optional spill path."""
        self.seal()
        out: CollectedOutput = {
            "text": b"".join(self._chunks).decode("utf-8", errors="replace"),
            "truncated": self._dropped,
        }
        if self._spill_file is not None:
            out["spillPath"] = self._spill_file
        return out


__all__ = [
    "OutputCollector",
    "SpillOptions",
    "default_spill_dir",
    "log_spill_failure",
]
