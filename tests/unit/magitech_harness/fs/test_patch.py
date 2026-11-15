"""Unit tests for patch parsing and hunk application."""

from __future__ import annotations

import pytest

from magitech_harness.fs.patch import (
    PatchParseError,
    _seek_sequence,
    apply_hunks,
    parse_patch,
)


def test_parse_patch_add_update_delete_and_move() -> None:
    """Parses wrapped patches with add, delete, update, move, and EOF hunks."""
    raw = """<<'EOF'
*** Begin Patch
*** Add File: notes/hello.txt
+alpha
+beta
*** Delete File: notes/old.txt
*** Update File: notes/main.txt
*** Move to: notes/renamed.txt
@@ def greet():
-    return "old"
+    return "new"
*** Update File: notes/tail.txt
@@
-last
+final
*** End of File
*** End Patch
EOF"""
    ops = parse_patch(raw)
    assert [op["kind"] for op in ops] == ["add", "delete", "update", "update"]
    assert ops[0] == {
        "kind": "add",
        "lineNumber": 3,
        "path": "notes/hello.txt",
        "content": "alpha\nbeta\n",
    }
    assert ops[1] == {
        "kind": "delete",
        "lineNumber": 6,
        "path": "notes/old.txt",
    }
    assert ops[2]["moveTo"] == "notes/renamed.txt"
    assert ops[2]["hunks"] == [
        {
            "lineNumber": 9,
            "context": "def greet():",
            "oldLines": ['    return "old"'],
            "newLines": ['    return "new"'],
            "isEndOfFile": False,
        }
    ]
    assert ops[3]["hunks"][0]["isEndOfFile"] is True


def test_parse_patch_errors() -> None:
    """Reports one-based original line numbers for syntax failures."""
    with pytest.raises(PatchParseError) as exc1:
        parse_patch("\nnot a begin marker\n*** End Patch")
    assert exc1.value.line_number == 2

    with pytest.raises(PatchParseError) as exc2:
        parse_patch("*** Begin Patch\n*** Update File: a.txt\n*** End Patch")
    assert exc2.value.line_number == 2

    with pytest.raises(PatchParseError) as exc3:
        parse_patch("*** Begin Patch\n*** Update File: a.txt\n@@\n*** End Patch")
    assert exc3.value.line_number == 4

    with pytest.raises(PatchParseError) as exc4:
        parse_patch("*** Begin Patch\n*** End of File\n*** End Patch")
    assert exc4.value.line_number == 2

    with pytest.raises(PatchParseError) as exc5:
        parse_patch("*** Begin Patch\n*** Add File: a.txt\nbad line\n*** End Patch")
    assert exc5.value.line_number == 3

    with pytest.raises(PatchParseError) as exc6:
        parse_patch(
            "*** Begin Patch\n*** Update File: a.txt\n@@\n-a\n+b\n*** End of File\noops\n*** End Patch"
        )
    assert exc6.value.line_number == 7

    with pytest.raises(PatchParseError) as exc7:
        parse_patch("*** Begin Patch\nno end")
    assert exc7.value.line_number == 2

    with pytest.raises(PatchParseError) as exc8:
        parse_patch(
            "*** Begin Patch\n*** Update File: a.txt\nbad-hunk-header\n*** End Patch"
        )
    assert exc8.value.line_number == 3

    with pytest.raises(PatchParseError) as exc9:
        parse_patch(
            "*** Begin Patch\n*** Update File: a.txt\n@@\n?invalid\n*** End Patch"
        )
    assert exc9.value.line_number == 4

    with pytest.raises(PatchParseError) as exc10:
        parse_patch("*** Begin Patch\n*** Update File: a.txt\n@@\n@@\n*** End Patch")
    assert exc10.value.line_number == 4

    with pytest.raises(PatchParseError) as exc11:
        parse_patch(
            "*** Begin Patch\n*** Update File: a.txt\n@@\n*** End of File\n*** End Patch"
        )
    assert exc11.value.line_number == 4

    with pytest.raises(PatchParseError) as exc12:
        parse_patch("*** Begin Patch\nstray line\n*** End Patch")
    assert exc12.value.line_number == 2

    with pytest.raises(PatchParseError) as exc13:
        parse_patch(
            "*** Begin Patch\n*** Delete File: a.txt\nstray inside delete\n*** End Patch"
        )
    assert exc13.value.line_number == 3


def test_parse_patch_whitespace_crlf_and_edge_cases() -> None:
    """Handles CRLF, blank separator lines, empty hunks at EOF, and full deletion."""
    raw = (
        "*** Begin Patch\r\n"
        "\r\n"
        "*** Delete File: dead.txt\r\r\n"
        "   \r\n"
        "*** Update File: a.txt  \r\n"
        "   \r\n"
        "@@ \r\n"
        " ctx\r\n"
        "\r\n"
        "-old\r\n"
        "+new\r\n"
        "*** End of File\r\n"
        "   \r\n"
        "@@ second context\r\n"
        "+tail\r\n"
        "*** End Patch"
    )
    ops = parse_patch(raw)
    assert ops[0]["kind"] == "delete"
    assert ops[1]["hunks"][0]["oldLines"] == ["ctx", "", "old"]
    assert ops[1]["hunks"][0]["newLines"] == ["ctx", "", "new"]
    assert len(ops[1]["hunks"]) == 2

    # Trailing whitespace-only trimmed + empty sequence fallback + trailing newline check
    empty_ctx = apply_hunks(
        "  first  \n\nsecond\n",
        [
            {
                "lineNumber": 1,
                "context": "",
                "oldLines": ["second"],
                "newLines": ["SECOND"],
                "isEndOfFile": False,
            }
        ],
    )
    assert empty_ctx == {"ok": True, "text": "  first  \n\nSECOND\n"}
    assert _seek_sequence(["a", "b"], [], 1, False) == 1
    assert apply_hunks(
        "hello\n",
        [
            {
                "lineNumber": 1,
                "oldLines": ["hello"],
                "newLines": ["world"],
                "isEndOfFile": False,
            }
        ],
    ) == {"ok": True, "text": "world\n"}

    # Trailing empty line preserved when document already ends with blank
    keep_trailing = apply_hunks(
        "keep\n",
        [
            {
                "lineNumber": 1,
                "oldLines": ["keep"],
                "newLines": ["done", ""],
                "isEndOfFile": False,
            }
        ],
    )
    assert keep_trailing == {"ok": True, "text": "done\n"}

    # Pattern longer than document -> failure
    long_fail = apply_hunks(
        "only\n",
        [
            {
                "lineNumber": 1,
                "oldLines": ["a", "b", "c"],
                "newLines": [],
                "isEndOfFile": False,
            }
        ],
    )
    assert long_fail["ok"] is False

    # Deleting all lines without trailing blank produces empty string; unmatched old with trailing blank
    deleted_all = apply_hunks(
        "sole",
        [{"lineNumber": 1, "oldLines": ["sole"], "newLines": [], "isEndOfFile": False}],
    )
    assert deleted_all == {"ok": True, "text": ""}

    # Leading-trimmed match without trailing blank fallback (newLines != blank)
    trimmed_hit = apply_hunks(
        "  indented\n  next\n",
        [
            {
                "lineNumber": 1,
                "oldLines": ["indented"],
                "newLines": ["REPLACED"],
                "isEndOfFile": False,
            }
        ],
    )
    assert trimmed_hit == {"ok": True, "text": "REPLACED\n  next\n"}

    # Trailing blank only on oldLines (newLines non-empty without blank) -> retries stripped pattern
    strip_old_only = apply_hunks(
        "keep",
        [
            {
                "lineNumber": 1,
                "oldLines": ["keep", ""],
                "newLines": ["done"],
                "isEndOfFile": True,
            }
        ],
    )
    assert strip_old_only == {"ok": True, "text": "done\n"}


def test_apply_hunks_passes_and_failures() -> None:
    """Applies exact, whitespace-trimmed, unicode-normalized, and append hunks."""
    before = "line one  \n\u201chello\u201d\nlast line\n"
    hunks = [
        {
            "lineNumber": 4,
            "oldLines": ["line one"],
            "newLines": ["line 1"],
            "isEndOfFile": False,
        },
        {
            "lineNumber": 7,
            "context": '"hello"',
            "oldLines": ["last line"],
            "newLines": ["end line"],
            "isEndOfFile": True,
        },
        {
            "lineNumber": 10,
            "oldLines": [],
            "newLines": ["extra"],
            "isEndOfFile": False,
        },
    ]
    res = apply_hunks(before, hunks)
    assert res == {
        "ok": True,
        "text": "line 1\n\u201chello\u201d\nend line\nextra\n",
    }

    empty_res = apply_hunks(
        "keep",
        [{"lineNumber": 2, "oldLines": [], "newLines": [], "isEndOfFile": False}],
    )
    assert empty_res == {"ok": True, "text": "keep\n"}

    trailing_blank = apply_hunks(
        "a\n",
        [
            {
                "lineNumber": 2,
                "oldLines": ["a", ""],
                "newLines": ["b", ""],
                "isEndOfFile": False,
            }
        ],
    )
    assert trailing_blank == {"ok": True, "text": "b\n"}

    ctx_fail = apply_hunks(
        "a\nb\n",
        [
            {
                "lineNumber": 2,
                "context": "missing",
                "oldLines": ["a"],
                "newLines": ["x"],
                "isEndOfFile": False,
            }
        ],
    )
    assert ctx_fail["ok"] is False
    assert ctx_fail["failure"]["stage"] == "context"

    old_fail = apply_hunks(
        "a\nb\n",
        [{"lineNumber": 3, "oldLines": ["z"], "newLines": ["y"], "isEndOfFile": False}],
    )
    assert old_fail["ok"] is False
    assert old_fail["failure"]["stage"] == "oldLines"


def test_parse_patch_env_id_trailing_space_and_out_of_order_hunks() -> None:
    """Rejects Environment ID headers, preserves trailing spaces, and sorts hunks."""
    with pytest.raises(PatchParseError, match="Environment ID") as exc_env:
        parse_patch("*** Begin Patch\n*** Environment ID: local\n*** End Patch")
    assert exc_env.value.line_number == 2

    ops = parse_patch(
        "*** Begin Patch\n"
        "*** Update File: space.txt\n"
        "@@\n"
        " keep  \n"
        "-old  \n"
        "+new  \n"
        "*** End Patch"
    )
    assert ops[0]["kind"] == "update"
    assert ops[0]["hunks"][0]["oldLines"] == ["keep  ", "old  "]
    assert ops[0]["hunks"][0]["newLines"] == ["keep  ", "new  "]

    # Out-of-order hunks resolved with non-monotonic offsets still apply cleanly
    applied = apply_hunks(
        "first\nsecond\n",
        [
            {
                "lineNumber": 1,
                "oldLines": [],
                "newLines": ["after-second"],
                "isEndOfFile": False,
            },
            {
                "lineNumber": 5,
                "oldLines": ["first"],
                "newLines": ["FIRST"],
                "isEndOfFile": False,
            },
        ],
    )
    assert applied == {"ok": True, "text": "FIRST\nsecond\nafter-second\n"}
