"""Pure patch parsing and LF-normalized hunk application."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Literal, NotRequired, TypedDict

_BEGIN = "*** Begin Patch"
_END = "*** End Patch"
_EOF = "*** End of File"
_MOVE = "*** Move to: "
_ADD = "*** Add File: "
_DELETE = "*** Delete File: "
_UPDATE = "*** Update File: "


class PatchHunk(TypedDict):
    """One contiguous patch hunk inside an update operation."""

    lineNumber: int
    context: NotRequired[str]
    oldLines: list[str]
    newLines: list[str]
    isEndOfFile: bool


class AddPatchOperation(TypedDict):
    """Patch operation creating a file with full initial contents."""

    kind: Literal["add"]
    lineNumber: int
    path: str
    contents: str


class DeletePatchOperation(TypedDict):
    """Patch operation removing an existing file."""

    kind: Literal["delete"]
    lineNumber: int
    path: str


class UpdatePatchOperation(TypedDict):
    """Patch operation updating (and optionally moving) an existing file."""

    kind: Literal["update"]
    lineNumber: int
    path: str
    moveTo: NotRequired[str]
    hunks: list[PatchHunk]


PatchOperation = AddPatchOperation | DeletePatchOperation | UpdatePatchOperation


class HunkApplyFailure(TypedDict):
    """Structured failure details for an unmatched patch hunk."""

    hunkLineNumber: int
    stage: Literal["context", "oldLines"]
    searchStart: int
    pattern: list[str]


class ApplyHunksSuccess(TypedDict):
    """Successful hunk application result."""

    ok: Literal[True]
    text: str


class ApplyHunksFailure(TypedDict):
    """Failed hunk application result naming the first unmatched hunk."""

    ok: Literal[False]
    failure: HunkApplyFailure


ApplyHunksResult = ApplyHunksSuccess | ApplyHunksFailure


class PatchParseError(ValueError):
    """Syntax failure with a one-based line number in the original input."""

    def __init__(self, line_number: int, reason: str) -> None:
        super().__init__(f"Invalid patch at line {line_number}: {reason}")
        self.line_number = line_number
        self.reason = reason


def _is_ws(ch: str) -> bool:
    return ch.isspace() or ch == "\ufeff"


def _trim(value: str) -> str:
    start = 0
    length = len(value)
    while start < length and _is_ws(value[start]):
        start += 1
    return _trim_end(value[start:])


def _trim_end(value: str) -> str:
    end = len(value)
    while end > 0 and _is_ws(value[end - 1]):
        end -= 1
    return value[:end]


def _leading_newlines(value: str) -> int:
    index = 0
    while index < len(value) and _is_ws(value[index]):
        index += 1
    return value[:index].count("\n")


def _patch_lines(patch: str) -> list[tuple[str, int]]:
    offset = _leading_newlines(patch)
    trimmed = _trim(patch)
    raw_lines = re.split(r"\r?\n", trimmed)
    lines = [(text, offset + idx + 1) for idx, text in enumerate(raw_lines)]
    if (
        len(lines) >= 4
        and lines[0][0] in ("<<EOF", "<<'EOF'", '<<"EOF"')
        and lines[-1][0].endswith("EOF")
    ):
        lines = lines[1:-1]
    start_text, start_num = lines[0]
    end_text, end_num = lines[-1]
    if _trim(start_text) != _BEGIN:
        raise PatchParseError(start_num, f"Expected '{_BEGIN}'")
    if _trim(end_text) != _END:
        raise PatchParseError(end_num, f"Expected '{_END}'")
    result: list[tuple[str, int]] = []
    last_idx = len(lines) - 1
    for idx, (text, num) in enumerate(lines):
        if idx < last_idx and text.endswith("\r"):
            result.append((text[:-1], num))
        else:
            result.append((text, num))
    return result


def _empty_hunk(hunk: PatchHunk) -> bool:
    return len(hunk["oldLines"]) == 0 and len(hunk["newLines"]) == 0


def _ensure_update(operation: PatchOperation | None, line_number: int) -> None:
    if operation is None or operation["kind"] != "update":
        return
    hunks = operation["hunks"]
    if not hunks:
        raise PatchParseError(
            operation["lineNumber"],
            f"Update for '{operation['path']}' requires at least one hunk",
        )
    if _empty_hunk(hunks[-1]):
        raise PatchParseError(
            line_number, "Update hunk must contain context, added, or removed lines"
        )


def _new_hunk(line_number: int) -> PatchHunk:
    return {
        "lineNumber": line_number,
        "oldLines": [],
        "newLines": [],
        "isEndOfFile": False,
    }


def _read_update(operation: UpdatePatchOperation, text_raw: str, line_num: int) -> None:
    text = _trim_end(text_raw)
    hunks = operation["hunks"]
    previous = hunks[-1] if hunks else None
    is_marker = text == "@@" or text.startswith("@@ ")
    if previous is not None and previous["isEndOfFile"]:
        if text == "":
            return
        if not is_marker:
            raise PatchParseError(line_num, f"Expected '@@' after '{_EOF}'")
    if previous is None and "moveTo" not in operation and text.startswith(_MOVE):
        operation["moveTo"] = text[len(_MOVE) :]
        return
    if is_marker:
        if previous is not None and _empty_hunk(previous):
            raise PatchParseError(
                line_num, "Update hunk must contain context, added, or removed lines"
            )
        hunk = _new_hunk(line_num)
        if len(text) > 3:
            hunk["context"] = text[3:]
        hunks.append(hunk)
        return
    if previous is None and text == "":
        return
    if previous is None:
        raise PatchParseError(line_num, f"Expected '@@' hunk header, got '{text}'")
    if text == _EOF:
        if _empty_hunk(previous):
            raise PatchParseError(line_num, f"Empty hunk before '{_EOF}'")
        previous["isEndOfFile"] = True
        return
    if text == "":
        previous["oldLines"].append("")
        previous["newLines"].append("")
        return
    prefix = text[0]
    body = text[1:]
    if prefix == " ":
        previous["oldLines"].append(body)
        previous["newLines"].append(body)
    elif prefix == "-":
        previous["oldLines"].append(body)
    elif prefix == "+":
        previous["newLines"].append(body)
    else:
        raise PatchParseError(line_num, f"Unexpected update line prefix '{prefix}'")


def parse_patch(patch: str) -> list[PatchOperation]:
    """Parse structured patch text into ordered add/delete/update operations."""
    lines = _patch_lines(patch)
    operations: list[PatchOperation] = []
    current: PatchOperation | None = None
    for text, line_num in lines[1:-1]:
        trimmed = _trim(text)
        if trimmed.startswith(_ADD):
            _ensure_update(current, line_num)
            current = {
                "kind": "add",
                "lineNumber": line_num,
                "path": trimmed[len(_ADD) :],
                "contents": "",
            }
            operations.append(current)
            continue
        if trimmed.startswith(_DELETE):
            _ensure_update(current, line_num)
            current = {
                "kind": "delete",
                "lineNumber": line_num,
                "path": trimmed[len(_DELETE) :],
            }
            operations.append(current)
            continue
        if trimmed.startswith(_UPDATE):
            _ensure_update(current, line_num)
            current = {
                "kind": "update",
                "lineNumber": line_num,
                "path": trimmed[len(_UPDATE) :],
                "hunks": [],
            }
            operations.append(current)
            continue
        if current is None:
            if text.strip() == "":
                continue
            raise PatchParseError(
                line_num, f"Unexpected line outside operation: '{text}'"
            )
        if current["kind"] == "add":
            if not text.startswith("+"):
                raise PatchParseError(line_num, "Add file line must start with '+'")
            current["contents"] += f"{text[1:]}\n"
            continue
        if current["kind"] == "delete":
            if text.strip() == "":
                continue
            raise PatchParseError(
                line_num, f"Unexpected line inside delete operation: '{text}'"
            )
        _read_update(current, text, line_num)
    _ensure_update(current, lines[-1][1])
    return operations


def _trim_ws_slice(value: str, leading: bool) -> str:
    start = 0
    end = len(value)
    if leading:
        while start < end and _is_ws(value[start]):
            start += 1
    while end > start and _is_ws(value[end - 1]):
        end -= 1
    return value[start:end]


_DASH_RE = re.compile(r"[\u2010-\u2015\u2212]")
_SQUOTE_RE = re.compile(r"[\u2018-\u201B]")
_DQUOTE_RE = re.compile(r"[\u201C-\u201F]")
_SPACE_RE = re.compile(r"[\u00A0\u2002-\u200A\u202F\u205F\u3000]")


def _normalize_punctuation(value: str) -> str:
    trimmed = _trim_ws_slice(value, True)
    trimmed = _DASH_RE.sub("-", trimmed)
    trimmed = _SQUOTE_RE.sub("'", trimmed)
    trimmed = _DQUOTE_RE.sub('"', trimmed)
    return _SPACE_RE.sub(" ", trimmed)


_MATCH_PASSES: tuple[Callable[[str], str], ...] = (
    lambda v: v,
    lambda v: _trim_ws_slice(v, False),
    lambda v: _trim_ws_slice(v, True),
    _normalize_punctuation,
)


def _seek_sequence(
    lines: Sequence[str], pattern: Sequence[str], start: int, eof: bool
) -> int | None:
    if len(pattern) == 0:
        return start
    if len(pattern) > len(lines):
        return None
    last = len(lines) - len(pattern)
    first = last if eof else start
    for normalize in _MATCH_PASSES:
        expected = [normalize(item) for item in pattern]
        for idx in range(first, last + 1):
            if all(
                normalize(lines[idx + offset]) == exp
                for offset, exp in enumerate(expected)
            ):
                return idx
    return None


def apply_hunks(before: str, hunks: Sequence[PatchHunk]) -> ApplyHunksResult:
    """Apply ordered hunks to LF-normalized text without I/O or mutation."""
    original = before.split("\n")
    if original and original[-1] == "":
        original.pop()
    replacements: list[tuple[int, int, Sequence[str]]] = []
    line_index = 0

    for hunk in hunks:
        context = hunk.get("context")
        if context is not None:
            pattern: list[str] = [context]
            found = _seek_sequence(original, pattern, line_index, False)
            if found is None:
                return {
                    "ok": False,
                    "failure": {
                        "hunkLineNumber": hunk["lineNumber"],
                        "stage": "context",
                        "searchStart": line_index,
                        "pattern": pattern,
                    },
                }
            line_index = found + 1
        old_lines = hunk["oldLines"]
        new_lines = hunk["newLines"]
        if len(old_lines) == 0:
            start = (
                len(original) - 1 if original and original[-1] == "" else len(original)
            )
            replacements.append((start, 0, new_lines))
            continue

        search_pattern = list(old_lines)
        applied_new = list(new_lines)
        found = _seek_sequence(
            original, search_pattern, line_index, hunk["isEndOfFile"]
        )
        if found is None and search_pattern and search_pattern[-1] == "":
            search_pattern = search_pattern[:-1]
            if applied_new and applied_new[-1] == "":
                applied_new = applied_new[:-1]
            found = _seek_sequence(
                original, search_pattern, line_index, hunk["isEndOfFile"]
            )
        if found is None:
            return {
                "ok": False,
                "failure": {
                    "hunkLineNumber": hunk["lineNumber"],
                    "stage": "oldLines",
                    "searchStart": line_index,
                    "pattern": list(old_lines),
                },
            }
        replacements.append((found, len(search_pattern), applied_new))
        line_index = found + len(search_pattern)

    updated = list(original)
    for start, old_len, repl_lines in reversed(replacements):
        updated[start : start + old_len] = list(repl_lines)

    if len(updated) == 0:
        return {"ok": True, "text": ""}
    if updated[-1] != "":
        updated.append("")
    return {"ok": True, "text": "\n".join(updated)}


parsePatch = parse_patch
applyHunks = apply_hunks

__all__ = [
    "ApplyHunksResult",
    "HunkApplyFailure",
    "PatchHunk",
    "PatchOperation",
    "PatchParseError",
    "applyHunks",
    "apply_hunks",
    "parsePatch",
    "parse_patch",
]
