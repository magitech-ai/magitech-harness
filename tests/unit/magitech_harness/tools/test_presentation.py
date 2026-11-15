"""Unit tests for tool presentation view definitions."""

from __future__ import annotations

from magitech_harness.tools.presentation import (
    DiffCallView,
    DiffResultView,
    FileDiff,
    FileLocation,
    GenericCallView,
    GenericResultView,
    ReadFileLine,
    ReadResultView,
    SearchFileMatches,
    SearchLineMatch,
    SearchMatchesResultView,
    SearchPathsResultView,
    TerminalCallView,
    TerminalResultView,
    WebFetchResultView,
    WebSearchResultView,
    WebSource,
)


def test_presentation_typed_views_round_trip() -> None:
    """Every ToolCallView and ToolResultView variant constructs as structured dicts."""
    loc: FileLocation = {"path": "a.py", "line": 3}
    diff: FileDiff = {"path": "a.py", "oldText": "x", "newText": "y"}
    create_diff: FileDiff = {"path": "new.py", "oldText": None, "newText": "pass"}

    call_generic: GenericCallView = {
        "card": "generic",
        "title": "Read a.py",
        "kind": "read",
        "rawInput": "a.py",
        "content": [{"type": "text", "text": "reading"}],
        "locations": [loc],
    }
    call_terminal: TerminalCallView = {
        "card": "terminal",
        "title": "ls",
        "description": "list files",
        "cwd": "/tmp",
    }
    call_diff: DiffCallView = {
        "card": "diff",
        "title": "Write new.py",
        "diffs": [create_diff],
        "locations": [{"path": "new.py"}],
    }
    assert call_generic["card"] == "generic"
    assert call_terminal["title"] == "ls"
    assert call_diff["diffs"][0]["oldText"] is None

    res_generic: GenericResultView = {
        "card": "generic",
        "title": "Done",
        "content": [{"type": "text", "text": "ok"}],
    }
    res_terminal: TerminalResultView = {
        "card": "terminal",
        "output": "file",
        "exitCode": 0,
    }
    res_terminal_sig: TerminalResultView = {
        "card": "terminal",
        "signal": "SIGTERM",
    }
    res_diff: DiffResultView = {"card": "diff", "diffs": [diff]}
    line_match: SearchLineMatch = {"lineNumber": 3, "line": "foo()"}
    file_match: SearchFileMatches = {"path": "a.py", "matches": [line_match]}
    res_matches: SearchMatchesResultView = {
        "card": "search",
        "shape": "matches",
        "files": [file_match],
        "truncated": False,
        "total": 1,
    }
    res_paths: SearchPathsResultView = {
        "card": "search",
        "shape": "paths",
        "paths": ["a.py"],
        "truncated": False,
        "total": 1,
    }
    read_line: ReadFileLine = {"number": 1, "text": "pass"}
    res_read: ReadResultView = {
        "card": "read",
        "path": "a.py",
        "offset": 1,
        "lines": [read_line],
        "totalLines": 1,
        "lang": "py",
        "content": [{"type": "text", "text": "pass"}],
    }
    source: WebSource = {
        "url": "https://example.com",
        "title": "T",
        "snippet": "s",
        "publishedAt": "2026-01-01T00:00:00Z",
    }
    res_web_s: WebSearchResultView = {
        "card": "web",
        "kind": "search",
        "sources": [source],
        "answer": "ans",
        "truncated": False,
    }
    res_web_f: WebFetchResultView = {
        "card": "web",
        "kind": "fetch",
        "url": "https://example.com",
        "statusCode": 200,
        "truncated": False,
    }
    assert res_generic["content"] == [{"type": "text", "text": "ok"}]
    assert res_terminal["exitCode"] == 0
    assert res_terminal_sig["signal"] == "SIGTERM"
    assert res_diff["diffs"] == [diff]
    assert res_matches["files"][0]["matches"][0]["lineNumber"] == 3
    assert res_paths["paths"] == ["a.py"]
    assert res_read["lines"][0]["number"] == 1
    assert res_web_s["sources"][0]["title"] == "T"
    assert res_web_f["statusCode"] == 200
