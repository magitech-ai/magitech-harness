"""Stable digests for instruction content and duplicate suppression."""

from __future__ import annotations

import hashlib


def instruction_digest(content: str) -> str:
    """SHA-1 of exact UTF-8 content (used by durable change records)."""
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def trimmed_instruction_digest(content: str) -> str:
    """SHA-1 of trimmed UTF-8 content (used by per-directory duplicate suppression)."""
    return hashlib.sha1(content.strip().encode("utf-8")).hexdigest()


__all__ = ["instruction_digest", "trimmed_instruction_digest"]
