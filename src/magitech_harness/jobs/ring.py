"""Bounded in-memory UTF-8 byte ring buffer with monotonic offset reads."""

from __future__ import annotations

from dataclasses import dataclass

from magitech_harness.jobs.types import JobAppendOptions, JobOutputChunk, JobOutputRead


def _utf8_tail(text: str, keep_bytes: int) -> tuple[str, int]:
    """Keep the suffix of *text* fitting inside *keep_bytes* on a UTF-8 char boundary."""
    if keep_bytes <= 0:
        return "", 0
    raw = text.encode("utf-8")
    start = len(raw) - keep_bytes
    while start < len(raw) and (raw[start] & 0xC0) == 0x80:
        start += 1
    tail = raw[start:]
    return tail.decode("utf-8", errors="replace"), len(tail)


@dataclass(slots=True)
class _RingEntry:
    at: int
    byte_len: int
    text: str
    channel: str | None
    gap_before: bool


class OutputRing:
    """Retention ring tracking monotonic whole-stream UTF-8 byte offsets."""

    def __init__(self) -> None:
        self._entries: list[_RingEntry] = []
        self._retained_bytes = 0
        self._total = 0

    @property
    def total(self) -> int:
        """Total UTF-8 bytes appended across the job's lifetime."""
        return self._total

    @property
    def earliest(self) -> int:
        """Earliest non-evicted byte offset still held in memory."""
        if not self._entries:
            return self._total
        return self._entries[0].at

    def append(
        self,
        text: str,
        options: JobAppendOptions | None,
        retain_bytes: int,
    ) -> bool:
        """Append *text* and evict oldest entries beyond *retain_bytes*."""
        if not text:
            return False
        raw_len = len(text.encode("utf-8"))
        channel = options.get("channel") if options is not None else None
        gap_before = bool(options.get("gapBefore")) if options is not None else False
        at = self._total
        self._total += raw_len
        self._entries.append(
            _RingEntry(
                at=at,
                byte_len=raw_len,
                text=text,
                channel=channel,
                gap_before=gap_before,
            )
        )
        self._retained_bytes += raw_len
        self.trim(retain_bytes)
        return True

    def trim(self, max_bytes: int) -> None:
        """Shrink retained byte window to at most *max_bytes* keeping latest suffix."""
        cap = max(0, max_bytes)
        while self._retained_bytes > cap and len(self._entries) > 1:
            dropped = self._entries.pop(0)
            self._retained_bytes -= dropped.byte_len
        if len(self._entries) == 1 and self._entries[0].byte_len > cap:
            head = self._entries[0]
            trimmed_text, actual_len = _utf8_tail(head.text, cap)
            if actual_len == 0:
                self._entries.pop(0)
                self._retained_bytes = 0
            else:
                head.at += head.byte_len - actual_len
                head.text = trimmed_text
                head.byte_len = actual_len
                head.gap_before = True
                self._retained_bytes = actual_len

    def read_from(self, from_offset: int) -> JobOutputRead:
        """Return retained chunks overlapping ``[from_offset, total)``."""
        chunks: list[JobOutputChunk] = []
        for entry in self._entries:
            if entry.at + entry.byte_len <= from_offset:
                continue
            chunk: JobOutputChunk = {"at": entry.at, "text": entry.text}
            if entry.channel is not None:
                chunk["channel"] = entry.channel
            if entry.gap_before:
                chunk["gapBefore"] = True
            chunks.append(chunk)
        return {
            "chunks": chunks,
            "next": self._total,
            "lossy": from_offset < self.earliest,
        }


__all__ = ["OutputRing"]
