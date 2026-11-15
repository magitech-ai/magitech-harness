"""Unit tests for OutputCollector bounded tail and best-effort spill files."""

from __future__ import annotations

from pathlib import Path

import pytest

from magitech_harness.subprocess.output import (
    OutputCollector,
    SpillOptions,
    default_spill_dir,
    log_spill_failure,
)


def test_output_collector_tail_only_and_incremental_reads() -> None:
    """Keeps byte-exact tail window and tracks whole-stream read offsets."""
    collector = OutputCollector(5, "stdout")
    collector.push(b"")
    collector.push(b"abc")
    assert collector.total_bytes == 3
    assert collector.read_from(0) == {"text": "abc", "nextOffset": 3, "lossy": False}
    collector.push(b"de")
    assert collector.read_from(3) == {"text": "de", "nextOffset": 5, "lossy": False}
    collector.push(b"fg")  # drops "ab" (head chunk <= excess) -> tail "cdefg"
    assert collector.read_from(1) == {"text": "cdefg", "nextOffset": 7, "lossy": True}
    collector.push(b"hij")  # trims head chunk -> tail "fghij"
    assert collector.snapshot() == (b"fghij", 10)
    collector.seal()
    assert collector.finalize() == {"text": "fghij", "truncated": True}


def test_output_collector_spilling_and_best_effort_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writes spill files lazily on overflow and degrades cleanly on cap/IO errors."""
    assert "magitech-subprocess-" in default_spill_dir()

    errors: list[str] = []

    class _Logger:
        def error(self, msg: str) -> None:
            errors.append(msg)

    reporter = log_spill_failure(_Logger())
    reporter(RuntimeError("disk full"), "stdout")
    assert len(errors) == 1 and "disk full" in errors[0]

    # Throwing reporter does not crash collector push
    spill_dir = str(tmp_path / "spills")
    c_boom = OutputCollector(
        2,
        "stdout",
        SpillOptions(
            1, spill_dir, lambda _exc, _lbl: (_ for _ in ()).throw(RuntimeError("rep"))
        ),
    )
    c_boom.push(b"abc")
    assert c_boom.finalize()["text"] == "bc"

    # Successful spill across overflow + read_from + finalize
    failures: list[tuple[BaseException, str]] = []
    c1 = OutputCollector(
        4,
        "stdout",
        SpillOptions(20, spill_dir, lambda exc, lbl: failures.append((exc, lbl))),
    )
    c1.push(b"")
    c1.push(b"123")
    c1.push(
        b"4567"
    )  # overflows 4B -> opens spill + replays prior "123" + writes "4567"
    r = c1.read_from(0)
    assert r["lossy"] is True
    assert "spillPath" in r
    spill_path = Path(r["spillPath"])
    assert spill_path.exists()
    assert spill_path.read_bytes() == b"1234567"
    c1.push(b"89")
    c1.push(b"")
    final = c1.finalize()
    assert final["truncated"] is True
    assert final["text"] == "6789"
    assert final["spillPath"] == str(spill_path)
    assert spill_path.read_bytes() == b"123456789"

    # Exceeding spill max_bytes discards and removes spill file
    c2 = OutputCollector(
        3,
        "stderr",
        SpillOptions(5, spill_dir, lambda exc, lbl: failures.append((exc, lbl))),
    )
    c2.push(b"1234")  # total 4 <= 5 -> spilled
    p2 = Path(c2.read_from(0)["spillPath"])
    assert p2.exists()
    c2.push(b"56")  # total 6 > 5 -> discarded & unlinked
    assert not p2.exists()
    assert "spillPath" not in c2.finalize()

    # Open/write OSError disables spilling and notifies on_failure without throwing
    import magitech_harness.subprocess.output as out_mod

    monkeypatch.setattr(
        out_mod.os,
        "open",
        lambda *_a, **_kw: (_ for _ in ()).throw(PermissionError("denied")),
    )
    c3 = OutputCollector(
        2,
        "stdout",
        SpillOptions(100, spill_dir, lambda exc, lbl: failures.append((exc, lbl))),
    )
    c3.push(b"abcd")
    assert len(failures) == 1
    assert c3.finalize()["text"] == "cd"
    monkeypatch.undo()

    # Close OSError during _discard_spill / seal and unlink OSError
    c4 = OutputCollector(
        2,
        "stdout",
        SpillOptions(10, spill_dir, lambda exc, lbl: failures.append((exc, lbl))),
    )
    c4.push(b"abcd")
    monkeypatch.setattr(
        out_mod.os,
        "close",
        lambda _fd: (_ for _ in ()).throw(OSError("close failed")),
    )
    monkeypatch.setattr(
        out_mod.os,
        "unlink",
        lambda _p: (_ for _ in ()).throw(OSError("unlink failed")),
    )
    c4.push(b"0123456789")  # triggers _discard_spill with close & unlink failures
    assert c4._spill_fd is not None
    c4.seal()  # close fails again -> drops spill_file and clears fd
    monkeypatch.undo()
    # Close actual leaked fd if still tracked from test above? No, _spill_fd was cleared on seal.
    assert failures  # Sanity check reporter received IO errors

    # Throwing reporter on spill IO failure does not propagate out of _spill_all
    monkeypatch.setattr(
        out_mod.os,
        "open",
        lambda *_a, **_kw: (_ for _ in ()).throw(OSError("open fail")),
    )
    c_io_boom = OutputCollector(
        2,
        "stdout",
        SpillOptions(
            64, spill_dir, lambda _exc, _lbl: (_ for _ in ()).throw(RuntimeError("rep"))
        ),
    )
    c_io_boom.push(b"abcde")
    assert c_io_boom.finalize()["text"] == "de"
    monkeypatch.undo()
