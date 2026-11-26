"""Model-facing ``apply_patch`` tool over the harness patch executor.

Ports ``packages/fs/tool-fs/src/{apply-patch,patch/*}.ts`` from the pinned dsh
source. The pure parser and hunk applier live in :mod:`magitech_harness.fs.patch`;
this module adds path resolution, read-only preflight, sequential commit, and
the compact model-facing rendering. The whole patch is checked before any
mutation; a commit stops at the first failure without rolling back earlier
operations, exactly as in dsh.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

from animas import Context

from magitech_harness.fs.patch import (
    PatchParseError,
    apply_hunks,
    parse_patch,
)
from magitech_harness.tools.fs import _tool_path_denied
from magitech_harness.tools.path_guard import PathEscapeError, confined_path
from magitech_harness.tools.schema import define_tool

APPLY_PATCH_SECTION_TEXT = (
    "Prefer apply_patch over separate edit calls for changes across files or "
    "separated regions. Give every hunk enough surrounding context lines to "
    "match one place in the current file. The whole patch is checked first "
    "(syntax, hunk matching, missing files) and a rejected patch changes "
    "nothing; once applying starts, operations run in order, stop at the first "
    "failure, and are not rolled back, so re-read affected files before "
    "retrying when needed."
)

APPLY_PATCH_DESCRIPTION = (
    "Create, update, delete, or move UTF-8 text files with one patch; prefer it "
    "over multiple edit calls when a change spans several files or separated "
    "regions."
)

_INPUT_DESCRIPTION = (
    "Patch text: start with *** Begin Patch and end with *** End Patch. "
    "Use *** Add File: path with + lines; *** Update File: path, optionally "
    "followed by *** Move to: path, with @@ or @@ context and space-prefixed "
    "context, + added, or - removed lines; *** End of File anchors the last "
    "hunk to the file end. Use *** Delete File: path to delete. Paths are "
    "relative to the Session working directory."
)


class _PatchFailure(Exception):
    """One preflight refusal with a model-facing message."""


class _PartialMove(Exception):
    """A move wrote its destination but could not remove the source."""


_State = dict[str, Any]
_Plan = dict[str, Any]
_Refusal = dict[str, Any]
_Outcome = dict[str, Any]


def _resolve(path_str: str, workspace_root: Path | str | None) -> Path:
    if workspace_root is None:
        return Path(path_str)
    try:
        return confined_path(workspace_root, path_str)
    except (PathEscapeError, OSError) as error:
        raise _PatchFailure(str(error)) from error


def _denied(
    ctx: Context | None,
    exec_view: Any,
    path_str: str,
    resolved: Path | None = None,
) -> str | None:
    if ctx is None:
        return None
    checked = str(resolved) if resolved is not None else path_str
    reason = _tool_path_denied(ctx, exec_view, checked)
    return reason[7:] if reason is not None and reason.startswith("Error: ") else reason


def _state_of(states: dict[Path, _State], path: Path) -> _State:
    state = states.get(path)
    if state is not None:
        return state
    try:
        info = path.stat()
        state = {
            "exists": True,
            "is_file": info.st_mode & 0o170000 == 0o100000,
            "content": None,
        }
    except OSError:
        state = {"exists": False, "is_file": False, "content": None}
    states[path] = state
    return state


def _content_of(state: _State, path: Path, display: str) -> str:
    if state["content"] is None:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise _PatchFailure(f'cannot patch "{display}": binary file') from error
        except OSError as error:
            raise _PatchFailure(f'cannot read "{display}": {error}') from error
        if "\0" in text:
            raise _PatchFailure(f'cannot patch "{display}": binary file')
        state["content"] = text.replace("\r\n", "\n")
    return cast(str, state["content"])


def _publish(state: _State, content: str | None) -> None:
    state["exists"] = content is not None
    state["is_file"] = content is not None
    state["content"] = content


def _preview(pattern: str) -> str:
    return pattern[:80] + ("…" if len(pattern) > 80 else "")


def _render_refusals(refusals: list[_Refusal]) -> str:
    lines = [f"Error: Patch rejected before any change ({len(refusals)} problems):"]
    for refusal in refusals:
        label = (
            f"- operation {refusal['index'] + 1} ({refusal['kind']} {refusal['path']}"
        )
        if refusal["destination"] is not None:
            label += f" -> {refusal['destination']}"
        lines.append(
            f"{label}, patch line {refusal['patch_line']}): {refusal['message']}"
        )
    return "\n".join(lines)


def _completed_lines(outcomes: list[_Outcome]) -> list[str]:
    lines: list[str] = []
    # dsh groups results as add, update (including moves), delete.
    for group in ("add", "update", "delete"):
        for outcome in outcomes:
            if outcome.get("status") != "completed":
                continue
            kind = "update" if outcome["kind"] == "move" else outcome["kind"]
            if kind != group:
                continue
            letter = {"add": "A", "delete": "D"}.get(kind, "M")
            lines.append(f"{letter} {outcome.get('destination') or outcome['path']}")
    return lines


def _render_commit(outcomes: list[_Outcome]) -> str:
    completed = sum(1 for item in outcomes if item.get("status") == "completed")
    if completed == len(outcomes):
        if not outcomes:
            return "Success. No operations in patch."
        return "\n".join(
            ["Success. Updated the following files:", *_completed_lines(outcomes)]
        )
    lines = [f"Patch stopped after {completed} of {len(outcomes)} operations."]
    if completed > 0:
        lines.append("Completed:")
        lines.extend(_completed_lines(outcomes))
    for status, title in (
        ("failed", "Failed:"),
        ("unknown", "Outcome unknown:"),
        ("skipped", "Not executed:"),
    ):
        failed = [item for item in outcomes if item.get("status") == status]
        if not failed:
            continue
        lines.append(title)
        for item in failed:
            label = f"- operation {item['index'] + 1} ({item['kind']} {item['path']}"
            if status == "skipped":
                lines.append(f"{label})")
                continue
            destination = (
                "" if item.get("destination") is None else f" -> {item['destination']}"
            )
            lines.append(
                f"{label}{destination}, patch line {item['patch_line']}): "
                f"{item.get('message', '')}"
            )
    return "\n".join(lines)


async def handle_apply_patch(
    args: dict[str, Any],
    *,
    ctx: Context | None = None,
    exec_view: Any = None,
    workspace_root: Path | str | None = None,
) -> str:
    """Preflight and commit one multi-file patch, rendering the dsh result."""
    raw = args.get("input")
    if not isinstance(raw, str) or not raw.strip():
        return "Error: 'input' argument is required."
    try:
        operations = parse_patch(raw)
    except PatchParseError as error:
        return f"Error: {error}"

    states: dict[Path, _State] = {}
    plans: list[_Plan] = []
    refusals: list[_Refusal] = []
    for index, operation in enumerate(operations):
        raw = cast(dict[str, Any], operation)
        kind = operation["kind"]
        path_str = operation["path"]
        move_to = cast(str | None, raw.get("moveTo")) if kind == "update" else None
        record_kind = "move" if move_to is not None else kind
        record: dict[str, Any] = {
            "index": index,
            "kind": record_kind,
            "path": path_str,
            "destination": move_to,
            "patch_line": operation["lineNumber"],
        }
        try:
            source = _resolve(path_str, workspace_root)
            reason = _denied(ctx, exec_view, path_str, source)
            if reason is not None:
                raise _PatchFailure(reason)
            destination = (
                _resolve(move_to, workspace_root) if move_to is not None else None
            )
            if destination is not None:
                reason = _denied(ctx, exec_view, str(move_to), destination)
                if reason is not None:
                    raise _PatchFailure(reason)
            if record_kind == "move" and source == destination:
                raise _PatchFailure(f'cannot move "{path_str}" to itself')
            raw_path = (
                Path(workspace_root) / path_str
                if workspace_root is not None
                else Path(path_str)
            )
            if record_kind in ("delete", "move") and raw_path.is_symlink():
                raise _PatchFailure(
                    f'cannot {record_kind} "{path_str}": path is a symbolic link — '
                    "operate on its target path instead"
                )
            source_state = _state_of(states, source)
            if kind == "add":
                if source_state["exists"] and not source_state["is_file"]:
                    raise _PatchFailure(f'cannot add "{path_str}": not a regular file')
                content: str | None = raw["content"]
                _publish(source_state, content)
                target = source
            elif kind == "delete":
                if not source_state["exists"]:
                    raise _PatchFailure(f'cannot delete "{path_str}": not found')
                if not source_state["is_file"]:
                    raise _PatchFailure(
                        f'cannot delete "{path_str}": not a regular file'
                    )
                content = None
                _publish(source_state, None)
                target = source
            else:
                if not source_state["exists"]:
                    raise _PatchFailure(f'cannot {record_kind} "{path_str}": not found')
                if not source_state["is_file"]:
                    raise _PatchFailure(
                        f'cannot {record_kind} "{path_str}": not a regular file'
                    )
                baseline = _content_of(source_state, source, path_str)
                result = apply_hunks(baseline, raw["hunks"])
                if not result["ok"]:
                    failure = result["failure"]
                    pattern = failure["pattern"][0] if failure["pattern"] else ""
                    raise _PatchFailure(
                        f'cannot patch "{path_str}": hunk at patch line '
                        f"{failure['hunkLineNumber']} ({failure['stage']}) does not "
                        f"match — {_preview(pattern)}"
                    )
                content = result["text"]
                if destination is None:
                    _publish(source_state, content)
                    target = source
                else:
                    destination_state = _state_of(states, destination)
                    if destination_state["exists"] and not destination_state["is_file"]:
                        raise _PatchFailure(
                            f'cannot move "{move_to}": not a regular file'
                        )
                    _publish(destination_state, content)
                    _publish(source_state, None)
                    target = destination
            plans.append(
                {
                    **record,
                    "source": source,
                    "target": target,
                    "content": content,
                    "baseline": baseline if kind in ("update",) else None,
                    "hunks": raw["hunks"] if kind == "update" else None,
                }
            )
        except _PatchFailure as error:
            refusals.append({**record, "message": str(error)})
    if refusals:
        return _render_refusals(refusals)

    outcomes: list[_Outcome] = [
        {
            "index": plan["index"],
            "kind": plan["kind"],
            "path": plan["path"],
            "destination": plan["destination"],
            "patch_line": plan["patch_line"],
            "status": "skipped",
        }
        for plan in plans
    ]

    async def commit(plan: _Plan) -> None:
        target = plan["target"]
        if plan["kind"] == "delete":
            await asyncio.to_thread(target.unlink)
            return

        content = plan["content"] or ""
        if plan["hunks"] is not None:
            try:
                current = await asyncio.to_thread(
                    plan["source"].read_text, encoding="utf-8"
                )
            except UnicodeDecodeError as error:
                raise _PatchFailure(
                    f'cannot patch "{plan["path"]}": binary file'
                ) from error
            normalized = current.replace("\r\n", "\n")
            if normalized != plan["baseline"]:
                raise _PatchFailure(
                    f'cannot patch "{plan["path"]}": file changed since it was read'
                )
            applied = apply_hunks(normalized, plan["hunks"])
            if not applied["ok"]:
                raise _PatchFailure(
                    f'cannot patch "{plan["path"]}": hunk no longer matches'
                )
            content = applied["text"]

        def write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        await asyncio.to_thread(write)
        if plan["kind"] == "move":
            try:
                await asyncio.to_thread(plan["source"].unlink)
            except OSError as error:
                raise _PartialMove(str(error)) from error

    # Sequential, no rollback: the executor commits one operation at a time.
    for index, plan in enumerate(plans):
        try:
            await commit(plan)
        except OSError as error:
            message = f'cannot {plan["kind"]} "{plan["path"]}": {error}'
            outcomes[index] = {
                **outcomes[index],
                "status": "failed",
                "message": message,
            }
            break
        except _PatchFailure as error:
            outcomes[index] = {
                **outcomes[index],
                "status": "failed",
                "message": str(error),
            }
            break
        except _PartialMove as error:
            destination_text = cast(str, plan["destination"] or "")
            outcomes[index] = {
                **outcomes[index],
                "status": "failed",
                "message": (
                    f'cannot move "{plan["path"]}": {error}\n'
                    f"  destination {destination_text} was written; "
                    f"source {plan['path']} remains"
                ),
                "partial_move": True,
            }
            break
        outcomes[index] = {**outcomes[index], "status": "completed"}
    if (
        outcomes
        and outcomes[0].get("status") == "failed"
        and not outcomes[0].get("partial_move")
        and all(item.get("status") == "skipped" for item in outcomes[1:])
    ):
        first = outcomes[0]
        return _render_refusals(
            [
                {
                    "index": first["index"],
                    "kind": first["kind"],
                    "path": first["path"],
                    "destination": first["destination"],
                    "patch_line": first["patch_line"],
                    "message": first.get("message", ""),
                }
            ]
        )
    return _render_commit(outcomes)


def register_apply_patch(
    ctx: Context, workspace_root: Path | str | None = None
) -> None:
    """Register the config-gated ``apply_patch`` tool and its guidance."""

    async def execute(args: dict[str, Any], exec_view: Any = None) -> str:
        return await handle_apply_patch(
            args, ctx=ctx, exec_view=exec_view, workspace_root=workspace_root
        )

    ctx.tools.register(
        define_tool(
            name="apply_patch",
            description=APPLY_PATCH_DESCRIPTION,
            parameters={
                "input": {
                    "type": "string",
                    "required": True,
                    "description": _INPUT_DESCRIPTION,
                }
            },
            execute=execute,
        )
    )
    system_prompt = ctx.get("systemPrompt", None, strict=False)
    if system_prompt is not None:
        system_prompt.section(
            {
                "name": "tool:apply_patch",
                "order": system_prompt.get_section_order("TOOL_APPLY_PATCH"),
                "text": APPLY_PATCH_SECTION_TEXT,
            }
        )


__all__ = [
    "APPLY_PATCH_DESCRIPTION",
    "APPLY_PATCH_SECTION_TEXT",
    "handle_apply_patch",
    "register_apply_patch",
]
