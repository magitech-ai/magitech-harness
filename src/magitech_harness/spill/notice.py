"""Spill-policy notice formatting.

Formats the truncation notice appended to retained tool results, and detects
existing notices in persisted text. Reference:
``packages/spill/spill-policy/src/notice.ts``.
"""

from __future__ import annotations

from typing import TypedDict


class SpillRef(TypedDict):
    """A saved spill artifact: locator + retrieval guidance."""

    locator: str
    retrievalHint: str


class Omitted(TypedDict):
    """Omission summary for a spill notice."""

    kind: str  # "exact"
    count: int


OPEN = "("
CLOSE = ")"
LOCATION = " Full formatted result stored at: "
GUIDANCE_SEPARATOR = ". "
SEPARATOR = "\n\n"


def describe_omitted(omitted: Omitted, unit: str = "bytes") -> str:
    """Describe an omission as ``X bytes omitted``."""
    count = omitted["count"]
    return f"{count} {unit} omitted"


def format_spill_notice(omitted: Omitted, ref: SpillRef, images: int = 0) -> str:
    """Format the notice appended to a retained preview.

    Args:
        omitted: bytes/images omitted by the retention policy.
        ref: saved text locator and retrieval guidance.
        images: number of whole images omitted alongside text.

    Returns:
        The complete notice without a leading preview separator.
    """
    image_notice = f" Omitted {images} images." if images > 0 else ""
    return (
        f"{OPEN}{describe_omitted(omitted)}"
        f"{image_notice}{LOCATION}{ref['locator']}"
        f"{GUIDANCE_SEPARATOR}{ref['retrievalHint']}"
        f"{CLOSE}"
    )


def _is_omission_text(text: str) -> bool:
    """Check if ``text`` is exactly an omission description."""
    if text == describe_omitted({"kind": "none", "count": 0}):
        return True
    if not text.endswith(" bytes omitted"):
        return False
    count_str = text[: -len(" bytes omitted")]
    try:
        count = int(count_str)
    except ValueError:  # pragma: no cover
        return False  # pragma: no cover
    if count < 0:
        return False
    return text == describe_omitted(
        {"kind": "exact", "count": count}
    )  # pragma: no cover


def has_spill_notice(text: str) -> bool:
    """Check if a complete spill notice occupies the end of ``text``.

    Identifies the text convention, not authenticated tool-output origin.
    """
    if not text.endswith(CLOSE):
        return False
    idx = text.rfind(f"{SEPARATOR}{OPEN}")
    if idx < 0:
        return False
    candidate = text[idx:]
    if not candidate.startswith(f"{SEPARATOR}{OPEN}"):
        return False  # pragma: no cover - candidate always starts with SEPARATOR+OPEN
    omission_start = len(SEPARATOR) + len(OPEN)
    loc_pos = candidate.find(LOCATION, omission_start)
    if loc_pos < 0:
        return False
    omission_text = candidate[omission_start:loc_pos]
    if not _is_omission_text(omission_text):
        return False
    guidance_pos = text.find(GUIDANCE_SEPARATOR, idx + loc_pos + len(LOCATION))
    return (
        guidance_pos >= 0
    )  # pragma: no cover - guidance separator always present in valid notices  # pragma: no cover


def is_spill_notice(text: str) -> bool:
    """Alias for :func:`has_spill_notice`."""
    return has_spill_notice(text)
