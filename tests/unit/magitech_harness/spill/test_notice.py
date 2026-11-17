"""Unit tests for spill-policy notice formatting."""

from __future__ import annotations

from magitech_harness.spill.notice import (
    _is_omission_text,
    describe_omitted,
    format_spill_notice,
    has_spill_notice,
    is_spill_notice,
)


def test_format_spill_notice_text_only() -> None:
    """Format a text-only spill notice."""
    ref = {
        "locator": "/tmp/spill/session-abc123/abc-def.txt",
        "retrievalHint": "Use read with offset/limit, or grep this path to search within it.",
    }
    result = format_spill_notice({"kind": "exact", "count": 5000}, ref, 0)
    assert "(5000 bytes omitted" in result
    assert "/tmp/spill/session-abc123/abc-def.txt" in result
    assert (
        "Use read with offset/limit, or grep this path to search within it." in result
    )


def test_format_spill_notice_with_images() -> None:
    """Format notice including image count."""
    ref = {
        "locator": "/tmp/spill/file.txt",
        "retrievalHint": "Use read with offset/limit, or grep this path to search within it.",
    }
    result = format_spill_notice({"kind": "exact", "count": 3000}, ref, 2)
    assert "(3000 bytes omitted" in result
    assert " Omitted 2 images." in result
    assert "/tmp/spill/file.txt" in result


def test_has_spill_notice_detects_notice() -> None:
    """Detect when text ends with a spill notice."""
    notice = format_spill_notice(
        {"kind": "exact", "count": 1000},
        {
            "locator": "/tmp/spill/f.txt",
            "retrievalHint": "Use read with offset/limit, or grep this path to search within it.",
        },
        0,
    )
    text = "some retained content\n\n" + notice
    assert has_spill_notice(text) is True


def test_has_spill_notice_no_notice() -> None:
    """No notice when text doesn't end with one."""
    assert has_spill_notice("just some text") is False
    assert has_spill_notice("") is False


def test_is_omission_text_negative_count() -> None:
    """_is_omission_text rejects negative count."""
    assert _is_omission_text("-5 bytes omitted") is False


def test_has_spill_notice_no_guidance_separator() -> None:
    """has_spill_notice returns False when GUIDANCE_SEPARATOR missing."""
    text = (
        "some text\n\n(100 bytes omitted. Full formatted result stored at: /tmp/f.txt)"
    )
    assert has_spill_notice(text) is False


def test_has_spill_notice_alias() -> None:
    """is_spill_notice is an alias for has_spill_notice."""
    notice = format_spill_notice(
        {"kind": "exact", "count": 100},
        {
            "locator": "/tmp/f.txt",
            "retrievalHint": "Use read with offset/limit, or grep this path to search within it.",
        },
        0,
    )
    text = "content\n\n" + notice
    assert is_spill_notice(text) is True
    assert is_spill_notice("plain text") is False


def test_has_spill_notice_no_location() -> None:
    """has_spill_notice returns False when no LOCATION string."""
    text = "some text\n\n(100 bytes omitted. Use read with offset/limit, or grep this path to search within it.)"
    assert has_spill_notice(text) is False


def test_has_spill_notice_bad_omission() -> None:
    """has_spill_notice returns False when omission text is invalid."""
    text = "some text\n\n(not a number bytes omitted. Full formatted result stored at: /tmp/f.txt. Use read with offset/limit, or grep this path to search within it.)"
    assert has_spill_notice(text) is False


def test_describe_omitted_none() -> None:
    """describe_omitted with kind 'none'."""
    assert describe_omitted({"kind": "none", "count": 0}) == "0 bytes omitted"


def test_is_omission_text_none() -> None:
    """_is_omission_text matches 'none' omission."""
    assert _is_omission_text("0 bytes omitted") is True


def test_has_spill_notice_no_notice_prefix() -> None:
    """has_spill_notice returns False when no \n\n( prefix."""
    assert (
        has_spill_notice(
            "(100 bytes omitted. Full formatted result stored at: /tmp/f.txt. Use read with offset/limit, or grep this path to search within it.)"
        )
        is False
    )
