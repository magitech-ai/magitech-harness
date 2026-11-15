"""UTF-8 safe bounded item and text stream retainers with exact omission counts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, TypedDict


class OmittedNone(TypedDict):
    """No items or bytes were omitted."""

    kind: Literal["none"]


class OmittedExact(TypedDict):
    """Exact number of omitted items or bytes."""

    kind: Literal["exact"]
    count: int


class OmittedUnknown(TypedDict):
    """More items or bytes were omitted without an exact count."""

    kind: Literal["unknown"]


Omitted = OmittedNone | OmittedExact | OmittedUnknown


class PushDecision(TypedDict):
    """Per-push retention verdict."""

    kept: bool
    truncated: bool


class ItemRetentionStrategy(TypedDict):
    """Head retention cap for discrete items."""

    kind: Literal["head"]
    maxItems: int


class RetainedItems[T](TypedDict):
    """Final snapshot from :class:`ItemRetainer`."""

    items: list[T]
    truncated: bool
    seen: int
    kept: int
    omitted: Omitted


class TextHeadStrategy(TypedDict):
    """Retain leading bytes up to *maxBytes*."""

    kind: Literal["head"]
    maxBytes: int


class TextTailStrategy(TypedDict):
    """Retain trailing bytes up to *maxBytes*."""

    kind: Literal["tail"]
    maxBytes: int


class TextHeadTailStrategy(TypedDict):
    """Retain prefix and suffix bytes, omitting the middle."""

    kind: Literal["headTail"]
    headBytes: int
    tailBytes: int


TextRetentionStrategy = TextHeadStrategy | TextTailStrategy | TextHeadTailStrategy


class RetainedText(TypedDict):
    """Final decoded snapshot from :class:`TextRetainer`."""

    text: str
    truncated: bool
    omittedBytes: Omitted


RetentionUnit = Literal["items", "bytes", "chars", "lines"]


class RetentionNotice(TypedDict):
    """Neutral input passed to :func:`format_retention_notice`."""

    scope: str
    strategy: Literal["head", "tail", "headTail"]
    unit: RetentionUnit
    limit: int | dict[str, int]
    kept: int
    omitted: Omitted


def _assert_budget(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def trim_trailing_partial_utf8(data: bytes) -> bytes:
    """Drop an incomplete trailing UTF-8 sequence at a prefix boundary."""
    if not data:
        return data
    idx = len(data) - 1
    while idx >= 0 and (data[idx] & 0xC0) == 0x80 and (len(data) - idx) <= 3:
        idx -= 1
    if idx < 0:
        return data
    lead = data[idx]
    expected = (
        1
        if lead < 0x80
        else 2
        if lead < 0xE0
        else 3
        if lead < 0xF0
        else 4
        if lead < 0xF8
        else 0
    )
    if expected == 0:
        return data
    return data[:idx] if (len(data) - idx) < expected else data


def trim_leading_continuation_utf8(data: bytes) -> bytes:
    """Drop leading UTF-8 continuation bytes ``10xxxxxx`` at a suffix boundary."""
    idx = 0
    length = len(data)
    while idx < length and (data[idx] & 0xC0) == 0x80:
        idx += 1
    return data[idx:]


class ItemRetainer[T]:
    """Bound an ordered stream of logical items while counting exact omissions."""

    def __init__(self, strategy: ItemRetentionStrategy) -> None:
        _assert_budget(strategy["maxItems"], "maxItems")
        self._max_items = strategy["maxItems"]
        self._items: list[T] = []
        self._seen = 0
        self._omitted = 0

    def push(self, item: T) -> PushDecision:
        """Offer one unit to the bounded collection."""
        self._seen += 1
        if len(self._items) < self._max_items:
            self._items.append(item)
            return {"kept": True, "truncated": False}
        self._omitted += 1
        return {"kept": False, "truncated": True}

    def finish(self) -> RetainedItems[T]:
        """Return kept items and exact omission count."""
        truncated = self._omitted > 0
        omitted: Omitted = (
            {"kind": "exact", "count": self._omitted} if truncated else {"kind": "none"}
        )
        return {
            "items": self._items,
            "truncated": truncated,
            "seen": self._seen,
            "kept": len(self._items),
            "omitted": omitted,
        }


class TextRetainer:
    """Byte-oriented UTF-8 text retainer supporting head, tail, and headTail strategies."""

    def __init__(self, strategy: TextRetentionStrategy) -> None:
        if strategy["kind"] == "head":
            _assert_budget(strategy["maxBytes"], "maxBytes")
            self._prefix_cap = strategy["maxBytes"]
            self._suffix_cap = 0
        elif strategy["kind"] == "tail":
            _assert_budget(strategy["maxBytes"], "maxBytes")
            self._prefix_cap = 0
            self._suffix_cap = strategy["maxBytes"]
        else:
            _assert_budget(strategy["headBytes"], "headBytes")
            _assert_budget(strategy["tailBytes"], "tailBytes")
            self._prefix_cap = strategy["headBytes"]
            self._suffix_cap = strategy["tailBytes"]

        self._prefix_chunks: list[bytes] = []
        self._prefix_held = 0
        self._suffix_chunks: list[bytes] = []
        self._suffix_held = 0
        self._total = 0

    def push(self, chunk: bytes | str) -> PushDecision:
        """Offer a bytes chunk or UTF-8 string chunk."""
        data = chunk.encode("utf-8") if isinstance(chunk, str) else bytes(chunk)
        before = self._total
        self._total += len(data)

        room = self._prefix_cap - self._prefix_held
        take = max(0, min(room, len(data)))
        if take > 0:
            self._prefix_chunks.append(data[:take])
            self._prefix_held += take

        if self._suffix_cap > 0:
            self._suffix_chunks.append(data)
            self._suffix_held += len(data)
            while (
                self._suffix_chunks
                and (self._suffix_held - len(self._suffix_chunks[0]))
                >= self._suffix_cap
            ):
                head = self._suffix_chunks.pop(0)
                self._suffix_held -= len(head)
            if self._suffix_chunks and self._suffix_held > self._suffix_cap:
                excess = self._suffix_held - self._suffix_cap
                self._suffix_chunks[0] = self._suffix_chunks[0][excess:]
                self._suffix_held -= excess

        dropped_now = self._omitted_at(self._total) > self._omitted_at(before)
        return {
            "kept": not dropped_now,
            "truncated": self._omitted_at(self._total) > 0,
        }

    def _omitted_at(self, total: int) -> int:
        prefix_len = min(total, self._prefix_cap)
        suffix_len = min(total - prefix_len, self._suffix_cap)
        return total - prefix_len - suffix_len

    def finish(self) -> RetainedText:
        """Decode retained prefix/suffix cleanly at UTF-8 boundaries and report omitted bytes."""
        prefix_len = min(self._total, self._prefix_cap)
        suffix_len = min(self._total - prefix_len, self._suffix_cap)
        prefix = b"".join(self._prefix_chunks)
        suffix_all = b"".join(self._suffix_chunks)
        suffix = suffix_all[self._suffix_held - suffix_len :]

        budget_omitted = self._omitted_at(self._total)
        if budget_omitted > 0:
            kept_prefix = trim_trailing_partial_utf8(prefix)
            kept_suffix = trim_leading_continuation_utf8(suffix)
            text = kept_prefix.decode("utf-8", errors="replace") + kept_suffix.decode(
                "utf-8", errors="replace"
            )
        else:
            kept_prefix = prefix
            kept_suffix = suffix
            text = (prefix + suffix).decode("utf-8", errors="replace")

        omitted = self._total - len(kept_prefix) - len(kept_suffix)
        truncated = omitted > 0
        omitted_meta: Omitted = (
            {"kind": "exact", "count": omitted} if truncated else {"kind": "none"}
        )
        return {"text": text, "truncated": truncated, "omittedBytes": omitted_meta}


def describe_omitted(omitted: Omitted, unit: RetentionUnit) -> str:
    """Format a standardized omission sentence clause without recovery instructions."""
    if omitted["kind"] == "none":
        return ""
    if omitted["kind"] == "exact":
        return f"Omitted {omitted['count']} {unit}."
    return f"More {unit} were omitted."


def format_retention_notice(
    notice: RetentionNotice,
    recovery: Callable[[RetentionNotice], str],
) -> str:
    """Join standardized omission clause and tool-supplied recovery guidance."""
    parts: Sequence[str] = (
        describe_omitted(notice["omitted"], notice["unit"]),
        recovery(notice),
    )
    return " ".join(part for part in parts if len(part) > 0)


def truncate_without_splitting_surrogate_pair(text: str, max_chars: int) -> str:
    """Cap *text* at *max_chars* without leaving an unpaired trailing high surrogate."""
    if len(text) <= max_chars:
        return text
    capped = text[:max_chars]
    if capped and 0xD800 <= ord(capped[-1]) <= 0xDBFF:
        return capped[:-1]
    return capped


describeOmitted = describe_omitted
formatRetentionNotice = format_retention_notice
truncateWithoutSplittingSurrogatePair = truncate_without_splitting_surrogate_pair

__all__ = [
    "ItemRetainer",
    "ItemRetentionStrategy",
    "Omitted",
    "PushDecision",
    "RetainedItems",
    "RetainedText",
    "RetentionNotice",
    "RetentionUnit",
    "TextRetainer",
    "TextRetentionStrategy",
    "describeOmitted",
    "describe_omitted",
    "formatRetentionNotice",
    "format_retention_notice",
    "trim_leading_continuation_utf8",
    "trim_trailing_partial_utf8",
    "truncateWithoutSplittingSurrogatePair",
    "truncate_without_splitting_surrogate_pair",
]
