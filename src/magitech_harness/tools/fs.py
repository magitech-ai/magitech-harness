"""Base filesystem tools: ``read_file``, ``write_file``, and ``edit_file``.

Ports dsh's ``@deepseek-ai/dsh-tool-fs`` at
``packages/fs/tool-fs/src/{read,write,edit}.ts``. The reference routes through
``ctx.fs``; this port operates on the local filesystem directly and confines
every path to the session workspace. Observable behavior — argument validation,
line pagination, unique-match edits, and the model-facing text — follows the
reference.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from animas import Context

from magitech_harness.tools.path_guard import PathEscapeError, confined_path
from magitech_harness.tools.schema import define_tool

DEFAULT_LIMIT = 2000
MAX_LIMIT = 2000
_BINARY_PROBE_BYTES = 8192

READ_SECTION_TEXT = (
    "Use the read_file tool — not shell commands like cat, sed, head, or tail — "
    "to inspect text files. Use offset and limit to read a line range or continue "
    "a large file. Results are capped at 2000 lines per call."
)
WRITE_SECTION_TEXT = (
    "Use the write_file tool to create or overwrite a text file. Parent "
    "directories are created automatically. Empty content is legitimate and "
    "writes an empty file; the path must not be blank."
)
EDIT_SECTION_TEXT = (
    "Use the edit_file tool to replace a unique target string in a file. The "
    "target must occur exactly once; use write_file to replace an entire file."
)


def _split_file_lines(text: str) -> list[str]:
    lines = text.split("\n")
    if text.endswith("\n"):
        lines.pop()
    return lines


def _validate_limit(limit: Any, default_limit: int, max_limit: int) -> int | None:
    """Resolve and validate the requested line limit."""
    if limit is None:
        return default_limit
    if not isinstance(limit, int) or isinstance(limit, bool):
        return default_limit
    if limit < 1:
        raise ValueError(f"Invalid limit '{limit}'. Must be 1 or greater.")
    if limit > max_limit:
        raise ValueError(f"Invalid limit '{limit}'. Must not exceed {max_limit}.")
    return limit


async def handle_read_file(
    args: dict[str, Any],
    *,
    default_limit: int = DEFAULT_LIMIT,
    max_limit: int = MAX_LIMIT,
    workspace_root: Path | str | None = None,
) -> str:
    """Read a file with 1-indexed offset/limit pagination.

    When no ``limit`` is supplied, ``default_limit`` applies; a ``limit`` above
    ``max_limit`` fails loud, enforcing the per-call read cap.
    """
    path_str = args.get("path")
    if not path_str:
        return "Error: 'path' argument is required."

    offset = args.get("offset")
    limit = _validate_limit(args.get("limit"), default_limit, max_limit)

    if offset is not None and offset < 1:
        return f"Error: Invalid offset '{offset}'. Must be 1 or greater."

    if workspace_root is not None:
        try:
            file_path = confined_path(workspace_root, path_str)
        except (PathEscapeError, OSError) as exc:
            return f"Error: {exc}"
    else:
        file_path = Path(path_str)
    if not file_path.exists():
        return f"Error: File not found: {path_str}"

    if file_path.is_dir():
        skill_md = file_path / "SKILL.md"
        if skill_md.is_file():
            file_path = skill_md
        else:
            return f"Error: Cannot read directory '{path_str}'"

    def _read_bytes() -> bytes:
        with file_path.open("rb") as handle:
            return handle.read()

    raw = await asyncio.to_thread(_read_bytes)
    if b"\x00" in raw[:_BINARY_PROBE_BYTES]:
        return f"Error: Binary file not supported: {path_str}"

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"Error: Binary file not supported: {path_str}"

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = _split_file_lines(text)
    total = len(lines)

    start = (offset or 1) - 1
    if start >= total > 0:
        return f"Error: Offset {offset} is beyond end of file ({total} lines total)"

    window = lines[start:]
    if limit is not None and len(window) > limit:
        window = window[:limit]

    return "\n".join(window)


async def handle_write_file(
    args: dict[str, Any], *, workspace_root: Path | str | None = None
) -> str:
    """Write content to a path, creating parent directories.

    A blank path is rejected; empty ``content`` legitimately writes an empty
    file. Paths are confined to the workspace root when one is supplied.
    """
    path_str = args.get("path")
    if not path_str or not path_str.strip():
        return "Error: 'path' argument is required."
    content = args.get("content", "")

    if workspace_root is not None:
        try:
            file_path = confined_path(workspace_root, path_str)
        except (PathEscapeError, OSError) as exc:
            return f"Error: {exc}"
    else:
        file_path = Path(path_str)

    def _write() -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    await asyncio.to_thread(_write)
    return f"Successfully wrote to {path_str}"


async def handle_edit_file(args: dict[str, Any]) -> str:
    """Replace a unique target string in a file.

    A blank path, empty target, or target equal to replacement are rejected; the
    target must match exactly once.
    """
    path_str = args.get("path")
    if not path_str or not path_str.strip():
        return "Error: 'path' argument is required."
    target = args.get("target")
    if target is None or target == "":
        return "Error: 'target' argument is required."
    replacement = args.get("replacement", "")

    if target == replacement:
        return f"Error: target and replacement must differ in {path_str}"

    file_path = Path(path_str)
    if not file_path.exists():
        return f"Error: File not found: {path_str}"

    def _perform_edit() -> str:
        content = file_path.read_text(encoding="utf-8")
        count = content.count(target)
        if count == 0:
            return f"Error: Target string not found in {path_str}"
        if count > 1:
            return (
                f"Error: Target string matched {count} times in {path_str}. "
                "Must match uniquely (exactly 1 occurrence)."
            )

        new_content = content.replace(target, replacement, 1)
        file_path.write_text(new_content, encoding="utf-8")
        return f"Successfully updated {path_str}"

    return await asyncio.to_thread(_perform_edit)


def _register_read(ctx: Context, cfg: dict[str, Any]) -> None:
    default_limit = int(cfg.get("default_limit", DEFAULT_LIMIT))
    max_limit = int(cfg.get("max_limit", MAX_LIMIT))
    configured_root = cfg.get("workspace_root")

    async def _execute(args: dict[str, Any]) -> str:
        # Resolve the root at call time so branch-worktree children confine to
        # their own workspace, not a captured parent root.
        root = configured_root or getattr(ctx, "workspace_dir", None)
        return await handle_read_file(
            args,
            default_limit=default_limit,
            max_limit=max_limit,
            workspace_root=root,
        )

    ctx.tools.register(
        define_tool(
            name="read_file",
            description="Read file contents with line offset and limit pagination.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path to file",
                    "required": True,
                },
                "offset": {"type": "integer", "description": "1-indexed starting line"},
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Maximum lines to read (default {default_limit}, "
                        f"max {max_limit})"
                    ),
                },
            },
            execute=_execute,
        )
    )


def _tool_path_denied(ctx: Context, exec_view: Any, path_str: str) -> str | None:
    """Return a denial reason when the resolved sandbox policy forbids a write."""
    policy_service = ctx.get("sandboxPolicy", None, strict=False)
    if policy_service is None:
        return None
    agent = exec_view.get("agent") if isinstance(exec_view, dict) else None
    session = getattr(agent, "session", None)
    if session is None and isinstance(exec_view, dict):
        session = exec_view.get("session")
    policy = policy_service.resolve(session)
    mode = str(policy.get("mode", "read-only"))
    if mode == "danger-full-access":
        return None
    try:
        path = Path(path_str).resolve(strict=False)
    except OSError:
        path = Path(path_str)
    roots = []
    if mode == "workspace-write":
        roots.extend(
            [
                str(Path(str(policy.get("workspaceRoot", ""))).resolve(strict=False)),
                *(
                    str(Path(str(extra)).resolve(strict=False))
                    for extra in policy.get("additionalWritableRoots") or ()
                ),
                "/tmp",
            ]
        )
    roots.append("/dev/null")
    for root in roots:
        try:
            path.relative_to(root)
            return None
        except ValueError:
            continue
    return f"[sandbox: file access denied under {mode} mode]"


def _register_write(ctx: Context, cfg: dict[str, Any]) -> None:
    configured_root = cfg.get("workspace_root")

    async def _execute(args: dict[str, Any], exec_view: Any = None) -> str:
        denied = _tool_path_denied(ctx, exec_view, str(args.get("path", "")))
        if denied is not None:
            return f"Error: {denied}"
        root = configured_root or getattr(ctx, "workspace_dir", None)
        return await handle_write_file(args, workspace_root=root)

    ctx.tools.register(
        define_tool(
            name="write_file",
            description="Write text content to a file, creating parent directories.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Target file path",
                    "required": True,
                },
                "content": {
                    "type": "string",
                    "description": "File content",
                    "required": True,
                },
            },
            execute=_execute,
        )
    )


def _register_edit(ctx: Context) -> None:
    async def _execute(args: dict[str, Any], exec_view: Any = None) -> str:
        denied = _tool_path_denied(ctx, exec_view, str(args.get("path", "")))
        if denied is not None:
            return f"Error: {denied}"
        return await handle_edit_file(args)

    ctx.tools.register(
        define_tool(
            name="edit_file",
            description="Replace target string with replacement string in a file.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Target file path",
                    "required": True,
                },
                "target": {
                    "type": "string",
                    "description": "Exact text to replace",
                    "required": True,
                },
                "replacement": {
                    "type": "string",
                    "description": "Replacement text",
                    "required": True,
                },
            },
            execute=_execute,
        )
    )


def _section(ctx: Context, name: str, order: str, text: str) -> None:
    system_prompt = ctx.get("systemPrompt", None, strict=False)
    if system_prompt is not None:
        system_prompt.section(
            {
                "name": name,
                "order": system_prompt.get_section_order(order),
                "text": text,
            }
        )


def apply(ctx: Context, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
    """Mount read_file, write_file, and edit_file onto ``ctx.tools``.

    Mirrors dsh's single ``tool-fs`` row. ``apply_patch`` is accepted for
    config-shape parity but must stay false: the patch tool is not ported.
    """
    cfg = {**(config or {}), **kwargs}
    if cfg.get("apply_patch", False):
        raise ValueError("tool-fs: apply_patch is not ported; set apply_patch: false")
    _register_read(ctx, cfg)
    _register_write(ctx, cfg)
    _register_edit(ctx)
    _section(ctx, "tool:read_file", "TOOL_READ", READ_SECTION_TEXT)
    _section(ctx, "tool:write_file", "TOOL_WRITE", WRITE_SECTION_TEXT)
    _section(ctx, "tool:edit_file", "TOOL_EDIT", EDIT_SECTION_TEXT)


apply.inject = ["tools"]  # type: ignore[attr-defined]

__all__ = [
    "DEFAULT_LIMIT",
    "EDIT_SECTION_TEXT",
    "MAX_LIMIT",
    "READ_SECTION_TEXT",
    "WRITE_SECTION_TEXT",
    "apply",
    "handle_edit_file",
    "handle_read_file",
    "handle_write_file",
]
