"""Unit tests for ItemRetainer, TextRetainer, and retention formatting helpers."""

from __future__ import annotations

import pytest

from magitech_harness.subprocess.retention import (
    ItemRetainer,
    TextRetainer,
    describe_omitted,
    format_retention_notice,
    trim_leading_continuation_utf8,
    trim_trailing_partial_utf8,
    truncate_without_splitting_surrogate_pair,
)


def test_item_retainer_head_and_validation() -> None:
    """Bounds items to maxItems and counts exact omissions."""
    with pytest.raises(ValueError, match="non-negative integer"):
        ItemRetainer({"kind": "head", "maxItems": -1})
    with pytest.raises(ValueError, match="non-negative integer"):
        ItemRetainer({"kind": "head", "maxItems": True})  # type: ignore[typeddict-item]

    ret = ItemRetainer[str]({"kind": "head", "maxItems": 2})
    assert ret.push("a") == {"kept": True, "truncated": False}
    assert ret.push("b") == {"kept": True, "truncated": False}
    assert ret.push("c") == {"kept": False, "truncated": True}
    assert ret.finish() == {
        "items": ["a", "b"],
        "truncated": True,
        "seen": 3,
        "kept": 2,
        "omitted": {"kind": "exact", "count": 1},
    }

    exact = ItemRetainer[int]({"kind": "head", "maxItems": 2})
    exact.push(1)
    assert exact.finish() == {
        "items": [1],
        "truncated": False,
        "seen": 1,
        "kept": 1,
        "omitted": {"kind": "none"},
    }


def test_text_retainer_head_tail_headtail_and_utf8_boundaries() -> None:
    """Handles head/tail/headTail retention and never splits multi-byte codepoints."""
    # Exact contiguous join preserves a multi-byte character across head/tail seam
    smiley = "\U0001f600"  # 4 bytes
    no_gap = TextRetainer({"kind": "headTail", "headBytes": 2, "tailBytes": 2})
    assert no_gap.push(smiley) == {"kept": True, "truncated": False}
    assert no_gap.finish() == {
        "text": smiley,
        "truncated": False,
        "omittedBytes": {"kind": "none"},
    }

    # Budget cuts inside UTF-8 sequences trim partials at prefix and continuation at suffix
    gap = TextRetainer({"kind": "headTail", "headBytes": 6, "tailBytes": 3})
    assert gap.push(f"abcd{smiley}xyzw".encode()) == {
        "kept": False,
        "truncated": True,
    }
    res = gap.finish()
    # "abcd" (4B) + partial 2B of smiley trimmed -> 4B prefix; "yzw" (3B) suffix
    assert res["text"] == "abcdyzw"
    assert res["truncated"] is True
    assert res["omittedBytes"] == {"kind": "exact", "count": 5}

    # Head-only & tail-only strategies including multi-chunk rolling suffix & single oversize chunk
    head = TextRetainer({"kind": "head", "maxBytes": 4})
    assert head.push("abcdef") == {"kept": False, "truncated": True}
    assert head.finish()["text"] == "abcd"

    tail = TextRetainer({"kind": "tail", "maxBytes": 4})
    assert tail.push("ab") == {"kept": True, "truncated": False}
    assert tail.push("cd") == {"kept": True, "truncated": False}
    assert tail.push("ef") == {"kept": False, "truncated": True}
    assert tail.push("ghijklmnop") == {"kept": False, "truncated": True}
    assert tail.finish()["text"] == "mnop"

    # Trim helpers edge cases (empty, <=3 continuations, >3 continuations, invalid lead, valid full multi-byte)
    assert trim_trailing_partial_utf8(b"") == b""
    assert trim_trailing_partial_utf8(b"\x80\x80\x80") == b"\x80\x80\x80"
    assert trim_trailing_partial_utf8(b"\x80\x80\x80\x80") == b"\x80\x80\x80\x80"
    assert trim_trailing_partial_utf8(b"abc\xff") == b"abc\xff"
    assert trim_trailing_partial_utf8("\u00e9".encode("utf-8")) == "\u00e9".encode(
        "utf-8"
    )
    assert trim_trailing_partial_utf8("\u20ac".encode("utf-8")[:2]) == b""
    assert trim_trailing_partial_utf8("\U0001f600".encode("utf-8")[:2]) == b""
    assert trim_leading_continuation_utf8(b"\x80\xbfabc") == b"abc"


def test_formatting_and_surrogate_helpers() -> None:
    """Formats omission wording consistently and prevents trailing high surrogates."""
    assert describe_omitted({"kind": "none"}, "items") == ""
    assert (
        describe_omitted({"kind": "exact", "count": 3}, "bytes") == "Omitted 3 bytes."
    )
    assert describe_omitted({"kind": "unknown"}, "lines") == "More lines were omitted."

    assert (
        format_retention_notice(
            {
                "scope": "grep",
                "strategy": "head",
                "unit": "items",
                "limit": 10,
                "kept": 10,
                "omitted": {"kind": "exact", "count": 5},
            },
            lambda n: f"Kept {n['kept']}.",
        )
        == "Omitted 5 items. Kept 10."
    )
    assert (
        format_retention_notice(
            {
                "scope": "grep",
                "strategy": "head",
                "unit": "items",
                "limit": 10,
                "kept": 2,
                "omitted": {"kind": "none"},
            },
            lambda _n: "",
        )
        == ""
    )

    assert truncate_without_splitting_surrogate_pair("abc", 5) == "abc"
    assert truncate_without_splitting_surrogate_pair("abcd", 2) == "ab"
    assert truncate_without_splitting_surrogate_pair("a\ud83d\ude00", 2) == "a"
