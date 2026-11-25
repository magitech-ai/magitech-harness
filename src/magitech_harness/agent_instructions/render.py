"""Deterministic instruction rendering within a byte budget."""

from __future__ import annotations

from typing import Any

from magitech_harness.agent_instructions.files import (
    GLOBAL_SCOPE_AGENTS,
    GLOBAL_SCOPE_USER,
    InstructionFile,
)

FRAME_OPEN = "<system-reminder>"
FRAME_CLOSE = "</system-reminder>"
BASELINE_INTRO = (
    "The following workspace instructions may be relevant to your work. Use them "
    "as guidance when applicable. More specific instructions take precedence over "
    "broader ones. They do not override system, developer, or direct user instructions."
)
REPLACEMENT_INTRO = (
    "This complete workspace instruction baseline replaces all earlier workspace "
    "instruction baselines. " + BASELINE_INTRO
)
EMPTY_REPLACEMENT_INTRO = (
    "This complete workspace instruction baseline replaces all earlier workspace "
    "instruction baselines. No workspace instructions are currently active."
)
COMPACT_INTRO = "Workspace instructions were omitted or truncated to fit the configured byte budget."
ADDITIONAL_GUIDANCE = (
    "Use them as guidance when relevant; more specific instructions take precedence. "
    "They do not override system, developer, or direct user instructions."
)


def escape_reminder(text: str) -> str:
    """Escape literal closing markers so nested content cannot break the frame."""
    return text.replace(FRAME_CLOSE, "<\\/system-reminder>")


def truncate_utf8(text: str, limit: int) -> str:
    """Truncate to at most ``limit`` bytes without splitting a code point."""
    encoded = text.encode("utf-8")[:limit]
    while encoded:
        try:
            return encoded.decode("utf-8")
        except UnicodeDecodeError:
            encoded = encoded[:-1]
    return ""


def _key(file: InstructionFile) -> str:
    return file.key or file.scope


def _baseline_section(file: InstructionFile) -> str:
    return f"Instructions from: {file.display_path}\n\n{escape_reminder(file.content)}"


def _additional_section(file: InstructionFile) -> str:
    if file.scope in (GLOBAL_SCOPE_USER, GLOBAL_SCOPE_AGENTS):
        applicability = "These user-global instructions apply to all work."
    else:
        applicability = f"These instructions apply to work under `{file.scope}`."
    return (
        f"Additional instructions from: {file.display_path}\n\n"
        f"{applicability}\n\n{ADDITIONAL_GUIDANCE}\n\n"
        f"{escape_reminder(file.content)}"
    )


def _frame(parts: list[str]) -> str:
    return f"{FRAME_OPEN}\n" + "\n\n".join(parts) + f"\n{FRAME_CLOSE}"


def _changes_for(
    files: list[InstructionFile],
    omitted: list[InstructionFile],
    truncated: list[tuple[InstructionFile, int, int]],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = [
        {
            "action": "set",
            "scope": _key(file),
            "path": file.display_path,
            "digest": file.digest,
        }
        for file in files
    ]
    for file in omitted:
        changes = [
            change
            for change in changes
            if not (
                change["scope"] == _key(file) and change["path"] == file.display_path
            )
        ]
    for file, _old, _new in truncated:
        changes = [
            change
            for change in changes
            if not (
                change["scope"] == _key(file) and change["path"] == file.display_path
            )
        ]
    return changes


def _budget_marker(
    max_bytes: int,
    omitted: list[InstructionFile],
    truncated: list[tuple[InstructionFile, int, int]],
) -> str:
    bits: list[str] = []
    if omitted:
        bits.append("omitted " + ", ".join(file.display_path for file in omitted))
    if truncated:
        bits.append(
            "; ".join(
                f"truncated {file.display_path} from {old} to {new} bytes"
                for file, old, new in truncated
            )
        )
    return f"Workspace instruction budget {max_bytes} bytes: " + "; ".join(bits)


def render_instruction_context(
    files: list[InstructionFile], max_bytes: int, *, baseline: bool = True
) -> tuple[str, list[dict[str, Any]]]:
    """Render files in loaded order, dropping broadest files before truncating."""
    if max_bytes <= 0:
        return "", [
            {
                "action": "remove",
                "scope": _key(file),
                "path": file.display_path,
                "digest": None,
            }
            for file in files
        ]
    intro = REPLACEMENT_INTRO if baseline else BASELINE_INTRO
    kept = list(files)  # loaded order: global, then project root -> cwd
    omitted: list[InstructionFile] = []
    truncated: list[tuple[InstructionFile, int, int]] = []
    while True:
        parts = [intro, *(_baseline_section(file) for file in kept)]
        text = _frame(parts)
        if len(text.encode("utf-8")) <= max_bytes:
            return text, _changes_for(files, omitted, truncated)
        if len(kept) > 1:
            omitted.insert(0, kept.pop(0))
            continue
        break

    if kept:
        file = kept[0]
        original = len(escape_reminder(file.content).encode("utf-8"))
        for use_compact in (False, True):
            head = COMPACT_INTRO if use_compact else intro
            marker = _budget_marker(max_bytes, omitted, truncated)
            content = escape_reminder(file.content)
            low, high, best = 0, len(content.encode("utf-8")), None
            while low <= high:
                middle = (low + high) // 2
                clipped = truncate_utf8(content, middle)
                text = _frame(
                    [
                        head,
                        marker,
                        f"Instructions from: {file.display_path}\n\n{clipped}",
                    ]
                )
                if len(text.encode("utf-8")) <= max_bytes:
                    best = (text, clipped)
                    low = middle + 1
                else:
                    high = middle - 1
            if best is not None:
                text, clipped = best
                truncated = [(file, original, len(clipped.encode("utf-8")))]
                return text, _changes_for(files, omitted, truncated)
        text = _frame([COMPACT_INTRO])
        if len(text.encode("utf-8")) > max_bytes:
            text = truncate_utf8(text, max_bytes)
        return text, _changes_for(files, omitted, [])

    text = _frame([COMPACT_INTRO])
    if len(text.encode("utf-8")) > max_bytes:
        text = truncate_utf8(text, max_bytes)
    return text, _changes_for(files, omitted, [])


def render_instruction_chains(
    files: list[InstructionFile], max_bytes: int, *, baseline: bool = True
) -> tuple[str, list[dict[str, Any]]]:
    """Use a single frame; split budgets only when the full set cannot fit."""
    single_text, single_changes = render_instruction_context(
        files, max_bytes, baseline=baseline
    )
    if max_bytes <= 0 or not files:
        return single_text, single_changes
    intro = REPLACEMENT_INTRO if baseline else BASELINE_INTRO
    full = _frame([intro, *(_baseline_section(file) for file in files)])
    if len(full.encode("utf-8")) <= max_bytes:
        return full, single_changes
    globals_ = [f for f in files if f.scope in (GLOBAL_SCOPE_USER, GLOBAL_SCOPE_AGENTS)]
    projects = [
        f for f in files if f.scope not in (GLOBAL_SCOPE_USER, GLOBAL_SCOPE_AGENTS)
    ]
    if not globals_ or not projects:
        return single_text, single_changes
    global_text, _ = render_instruction_context(globals_, max_bytes, baseline=baseline)
    project_text, _ = render_instruction_context(projects, max_bytes, baseline=baseline)
    sections: list[str] = []
    for text in (global_text, project_text):
        inner = text[len(FRAME_OPEN) + 1 : text.rfind(FRAME_CLOSE)].strip()
        parts = inner.split("\n\n")
        sections.extend(parts[1:])
    merged = _frame([COMPACT_INTRO, *sections])
    if len(merged.encode("utf-8")) > 2 * max_bytes:
        return single_text, single_changes
    return merged, single_changes


def render_changes(
    changes: list[dict[str, Any]],
    files: list[InstructionFile],
    max_bytes: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Render change-specific sections for a delta, not the whole baseline."""
    by_key = {_key(file): file for file in files}
    parts: list[str] = []
    for change in changes:
        scope = change.get("scope")
        file = by_key.get(scope) if isinstance(scope, str) else None
        action = change.get("action")
        path = change.get("path", "unknown")
        if action == "remove" or file is None:
            parts.append(
                f"Instructions removed: {path}\n\n"
                "The previously loaded instructions from this file no longer apply."
            )
        elif action == "replace":
            parts.append(
                f"Updated instructions from: {path}\n\n"
                "This file changed after it was loaded. Use the following content "
                f"instead of the previously loaded instructions from this file.\n\n"
                f"{escape_reminder(file.content)}"
            )
        elif action == "set":
            parts.append(_additional_section(file))
    if not parts:
        return "", changes
    text = _frame(parts)
    if len(text.encode("utf-8")) > max_bytes:
        text = truncate_utf8(text, max_bytes)
    return text, changes


__all__ = [
    "ADDITIONAL_GUIDANCE",
    "BASELINE_INTRO",
    "COMPACT_INTRO",
    "FRAME_CLOSE",
    "FRAME_OPEN",
    "REPLACEMENT_INTRO",
    "escape_reminder",
    "render_changes",
    "render_instruction_chains",
    "render_instruction_context",
    "truncate_utf8",
]
