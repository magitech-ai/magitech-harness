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


def _section(file: InstructionFile) -> str:
    return f"Instructions from: {file.display_path}\n\n{escape_reminder(file.content)}"


def _additional(file: InstructionFile) -> str:
    if file.scope in (GLOBAL_SCOPE_USER, GLOBAL_SCOPE_AGENTS):
        applicability = "These user-global instructions apply to all work."
    else:
        applicability = f"These instructions apply to work under `{file.scope}`."
    return (
        f"Additional instructions from: {file.display_path}\n\n"
        f"{applicability}\n\n{ADDITIONAL_GUIDANCE}\n\n"
        f"{escape_reminder(file.content)}"
    )


def render_instruction_context(
    files: list[InstructionFile], max_bytes: int, *, baseline: bool = True
) -> tuple[str, list[dict[str, Any]]]:
    """Render all files into one escaped frame, dropping broadest ones to fit."""
    changes: list[dict[str, Any]] = [
        {
            "action": "set",
            "scope": file.scope,
            "path": file.display_path,
            "digest": file.digest,
        }
        for file in files
    ]
    if max_bytes <= 0:
        return "", [
            {**change, "action": "remove", "digest": None} for change in changes
        ]
    intro = REPLACEMENT_INTRO if baseline else BASELINE_INTRO
    ordered = list(files)
    while True:
        sections = [_section(file) for file in ordered]
        parts = [intro, *sections]
        text = f"{FRAME_OPEN}\n" + "\n\n".join(parts) + f"\n{FRAME_CLOSE}"
        if len(text.encode("utf-8")) <= max_bytes:
            return text, changes
        if not ordered:
            compact = f"{FRAME_OPEN}\n{COMPACT_INTRO}\n{FRAME_CLOSE}"
            return compact[:max_bytes], changes
        # Broadest first: global scopes, then shallowest project directories.
        ordered.sort(
            key=lambda file: (
                0 if file.scope in (GLOBAL_SCOPE_USER, GLOBAL_SCOPE_AGENTS) else 1,
                -len(file.scope),
            )
        )
        dropped = ordered.pop(0)
        changes = [
            {**change, "action": "remove", "digest": None}
            if change["scope"] == dropped.scope
            and change["path"] == dropped.display_path
            else change
            for change in changes
        ]


def render_instruction_chains(
    files: list[InstructionFile], max_bytes: int, *, baseline: bool = True
) -> tuple[str, list[dict[str, Any]]]:
    """Budget the user-global and project chains separately, then merge."""
    single_text, single_changes = render_instruction_context(
        files, max_bytes, baseline=baseline
    )
    if max_bytes <= 0:
        return single_text, single_changes
    globals_ = [
        file for file in files if file.scope in (GLOBAL_SCOPE_USER, GLOBAL_SCOPE_AGENTS)
    ]
    projects = [
        file
        for file in files
        if file.scope not in (GLOBAL_SCOPE_USER, GLOBAL_SCOPE_AGENTS)
    ]
    if not globals_ or not projects:
        return single_text, single_changes
    global_text, global_changes = render_instruction_context(
        globals_, max_bytes, baseline=baseline
    )
    project_text, project_changes = render_instruction_context(
        projects, max_bytes, baseline=baseline
    )
    merged = global_text + "\n\n" + project_text
    if len(merged.encode("utf-8")) > 2 * max_bytes:
        return single_text, single_changes
    return merged, [*global_changes, *project_changes]


__all__ = [
    "ADDITIONAL_GUIDANCE",
    "BASELINE_INTRO",
    "COMPACT_INTRO",
    "FRAME_CLOSE",
    "FRAME_OPEN",
    "REPLACEMENT_INTRO",
    "escape_reminder",
    "render_instruction_chains",
    "render_instruction_context",
]
