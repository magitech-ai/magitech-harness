"""Durable instruction state reconstructed from ``user/message`` sources."""

from __future__ import annotations

from typing import Any

from magitech_harness.agent_instructions.files import InstructionFile
from magitech_harness.llm.message import create_user_message

SOURCE_KIND = "agent-instructions"


def instruction_source(
    text: str,
    changes: list[dict[str, Any]],
    *,
    baseline: bool = False,
    identity: str | None = None,
) -> dict[str, Any]:
    """Build the typed source for one instruction user message."""
    return {
        "kind": SOURCE_KIND,
        "form": "instructions",
        **({"baseline": True} if baseline else {}),
        **({} if identity is None else {"baselineIdentity": identity}),
        "changes": changes,
    }


def agent_instructions_message(
    text: str,
    changes: list[dict[str, Any]],
    *,
    baseline: bool = False,
    identity: str | None = None,
) -> dict[str, Any]:
    """One durable user message carrying an instruction baseline or delta."""
    return create_user_message(
        content=[{"type": "text", "text": text}],
        source=instruction_source(text, changes, baseline=baseline, identity=identity),
    )


def instruction_changes(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fold all visible instruction changes, last write wins per logical scope."""
    visible: dict[str, dict[str, Any]] = {}
    for message in messages:
        source = message.get("source")
        if not isinstance(source, dict) or source.get("kind") != SOURCE_KIND:
            continue
        changes = source.get("changes")
        if not isinstance(changes, list):
            continue
        for change in changes:
            if not isinstance(change, dict):
                continue
            action = change.get("action")
            scope = change.get("scope")
            path = change.get("path")
            if action not in ("set", "replace", "remove"):
                continue
            if not isinstance(scope, str) or not isinstance(path, str):
                continue
            digest = change.get("digest")
            if digest is not None and not isinstance(digest, str):
                continue
            if action == "remove":
                visible.pop(scope, None)
            else:
                visible[scope] = dict(change)
    return visible


def visible_baseline(
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Latest baseline-flagged instruction source, or None."""
    for message in reversed(messages):
        source = message.get("source")
        if (
            isinstance(source, dict)
            and source.get("kind") == SOURCE_KIND
            and source.get("baseline") is True
        ):
            return source
    return None


def baseline_instruction_state(
    files: list[InstructionFile],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Convert retained files into durable changes and a version cache."""
    changes = [
        {
            "action": "set",
            "scope": file.key or file.scope,
            "path": file.display_path,
            "digest": file.digest,
        }
        for file in files
    ]
    versions = {
        (file.key or file.scope): {
            "path": file.display_path,
            "digest": file.digest,
            "trimmedDigest": file.trimmed_digest,
        }
        for file in files
    }
    return changes, versions


def reconcile(
    messages: list[dict[str, Any]], files: list[InstructionFile]
) -> list[dict[str, Any]]:
    """Compute replace/remove changes against the currently visible scopes."""
    current = instruction_changes(messages)
    desired = {(file.key or file.scope): file for file in files}
    changes: list[dict[str, Any]] = []
    for scope, previous in current.items():
        file = desired.get(scope)
        if file is None:
            changes.append(
                {
                    "action": "remove",
                    "scope": scope,
                    "path": previous["path"],
                    "digest": None,
                }
            )
    for scope, file in desired.items():
        prior = current.get(scope)
        if prior is None or prior.get("digest") != file.digest:
            changes.append(
                {
                    "action": "set" if prior is None else "replace",
                    "scope": scope,
                    "path": file.display_path,
                    "digest": file.digest,
                }
            )
    return changes


__all__ = [
    "SOURCE_KIND",
    "agent_instructions_message",
    "baseline_instruction_state",
    "instruction_changes",
    "instruction_source",
    "reconcile",
    "visible_baseline",
]
