"""Unit tests for magitech_harness.jobs.ring."""

from __future__ import annotations

from magitech_harness.jobs.ring import OutputRing


def test_output_ring_empty_and_empty_append() -> None:
    ring = OutputRing()
    assert ring.total == 0
    assert ring.earliest == 0
    assert ring.append("", None, 100) is False
    assert ring.read_from(0) == {"chunks": [], "next": 0, "lossy": False}


def test_output_ring_append_read_overlap_and_channels() -> None:
    ring = OutputRing()
    assert ring.append("hello", {"channel": "stdout"}, 100) is True
    assert ring.append(" world", {"channel": "stderr", "gapBefore": True}, 100) is True
    assert ring.total == 11
    assert ring.earliest == 0

    full = ring.read_from(0)
    assert full == {
        "chunks": [
            {"at": 0, "text": "hello", "channel": "stdout"},
            {"at": 5, "text": " world", "channel": "stderr", "gapBefore": True},
        ],
        "next": 11,
        "lossy": False,
    }

    # Reading from inside a chunk returns the whole overlapping chunk
    mid = ring.read_from(2)
    assert mid["chunks"] == full["chunks"]
    assert mid["next"] == 11
    assert mid["lossy"] is False

    tail = ring.read_from(5)
    assert tail["chunks"] == [
        {"at": 5, "text": " world", "channel": "stderr", "gapBefore": True}
    ]

    at_end = ring.read_from(11)
    assert at_end == {"chunks": [], "next": 11, "lossy": False}


def test_output_ring_eviction_and_utf8_safe_single_chunk_trim() -> None:
    ring = OutputRing()
    ring.append("alpha", {"channel": "stdout"}, 100)
    ring.append("beta", {"channel": "stderr"}, 6)
    assert ring.earliest == 5
    res = ring.read_from(0)
    assert res == {
        "chunks": [{"at": 5, "text": "beta", "channel": "stderr"}],
        "next": 9,
        "lossy": True,
    }

    # Single oversized chunk with multi-byte UTF-8 boundary ('€' = 3 bytes)
    ring2 = OutputRing()
    ring2.append("a€b", {"channel": "stdout"}, 3)
    # "a€b" is 5 bytes (1 + 3 + 1); keeping <= 3 bytes skips inside '€' -> "b" (1 byte at offset 4)
    assert ring2.earliest == 4
    out = ring2.read_from(0)
    assert out == {
        "chunks": [{"at": 4, "text": "b", "channel": "stdout", "gapBefore": True}],
        "next": 5,
        "lossy": True,
    }

    # Trim to 0 clears the last chunk
    ring2.trim(0)
    assert ring2.earliest == 5
    assert ring2.read_from(0) == {"chunks": [], "next": 5, "lossy": True}

    # Append with cap=0 drops the single chunk via _utf8_tail keep_bytes<=0
    ring3 = OutputRing()
    assert ring3.append("abc", None, 0) is True
    assert ring3.total == 3
    assert ring3.earliest == 3

    # Read a retained chunk with channel=None
    ring4 = OutputRing()
    assert ring4.append("plain", None, 100) is True
    assert ring4.read_from(0) == {
        "chunks": [{"at": 0, "text": "plain"}],
        "next": 5,
        "lossy": False,
    }
