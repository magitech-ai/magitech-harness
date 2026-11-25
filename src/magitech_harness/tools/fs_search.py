"""Base filesystem search tools: ``glob`` and ``grep``.

Ports dsh's ``@deepseek-ai/dsh-tool-fs-search`` at
``packages/fs/tool-fs-search/src/{glob,grep}.ts``. The reference drives
ripgrep through ``ctx.subprocess``; this port uses ``pathlib`` and ``re`` so the
wheel stays self-contained and cross-platform. The observable contract — path
discovery including hidden and ignored files, modification-time order, regex
content search with line numbers grouped by file, result caps with a truncation
note, and workspace confinement — follows the reference.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from animas import Context

from magitech_harness.tools.path_guard import PathEscapeError, confined_path
from magitech_harness.tools.schema import define_tool

DEFAULT_MAX_RESULTS = 100
DEFAULT_MAX_MATCHES = 250
DEFAULT_MAX_LINE_BYTES = 2000
_VCS_DIRS = {".git", ".svn", ".hg", ".bzr", ".jj", ".sl"}

GLOB_SECTION_TEXT = (
    "Use the glob tool — not shell find — to discover files by path pattern. "
    "A pattern with no slash matches the basename at any depth; include a "
    "separator to anchor the depth."
)
GREP_SECTION_TEXT = (
    "Use the grep tool — not shell grep or rg — to search file contents. "
    "Use read_file on a matched file when you need surrounding context."
)


def _resolve_root(raw: str | None, workspace_root: Path | str | None) -> Path:
    base = (
        Path(workspace_root).resolve(strict=True)
        if workspace_root
        else Path.cwd().resolve()
    )
    if raw is None or not raw.strip():
        return base
    try:
        return confined_path(base, raw)
    except (PathEscapeError, OSError):  # pragma: no cover - defensive path escape
        return base


def _relative(root: Path, target: Path) -> Path:
    try:
        return target.relative_to(root)
    except ValueError:
        return target


def handle_glob(
    args: dict[str, Any],
    *,
    workspace_root: Path | str | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> str:
    """Find files matching a glob pattern under the workspace root."""
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        return "Error: 'pattern' argument is required and must be a non-empty string."
    root = _resolve_root(args.get("path"), workspace_root)

    base = root if root.is_dir() else root.parent
    matched: list[Path] = []
    try:
        candidates = base.glob(pattern) if "/" in pattern else base.rglob(pattern)
        for candidate in candidates:
            try:
                if not candidate.is_file():
                    continue
                if any(part in _VCS_DIRS for part in candidate.parts):
                    continue
                resolved = candidate.resolve(strict=False)
                confined_path(workspace_root or base, str(candidate))
            except (PathEscapeError, OSError):  # pragma: no cover - defensive
                continue
            matched.append(resolved)
    except (ValueError, OSError) as exc:  # pragma: no cover - rare pathlib error
        return f"Error: invalid glob pattern: {exc}"

    matched.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    rel = [_relative(root, p).as_posix() for p in matched]
    if not rel:
        return "No files found"
    if len(rel) <= max_results:
        return "\n".join(rel)
    kept = rel[:max_results]
    return (
        "\n".join(kept)
        + f"\n(Showing {len(kept)} of {len(rel)} paths. Full sorted result "
        "was limited by maxResults.)"
    )


def _iter_files(root: Path, include: str | None) -> list[Path]:
    files: list[Path] = []
    for candidate in root.rglob("*"):
        try:
            if not candidate.is_file():
                continue
            if any(part in _VCS_DIRS for part in candidate.parts):
                continue
            if include and not _fnmatch_path(candidate, include):
                continue
            confined_path(root, str(candidate))
            files.append(candidate)
        except (
            PathEscapeError,
            OSError,
            ValueError,
        ):  # pragma: no cover - defensive fs error
            continue
    return files


def _fnmatch_path(path: Path, include: str) -> bool:
    if "/" in include:
        return fnmatch.fnmatch(str(path), f"*{include}")
    return fnmatch.fnmatch(path.name, include)


def handle_grep(
    args: dict[str, Any],
    *,
    workspace_root: Path | str | None = None,
    max_matches: int = DEFAULT_MAX_MATCHES,
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
) -> str:
    """Search file contents by regex under the workspace root."""
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or pattern == "":
        return "Error: 'pattern' argument is required."
    include = args.get("include")
    if include is not None and (not isinstance(include, str) or not include.strip()):
        return "Error: 'include' argument must be a non-empty string."
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"Error: invalid regular expression: {exc}"

    root = _resolve_root(args.get("path"), workspace_root)
    files = [root] if root.is_file() else _iter_files(root, include)

    matches: list[tuple[str, int, str]] = []
    for file in files:
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover - defensive fs read error
            continue
        rel = _relative(root, file).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                preview = line
                if len(preview.encode("utf-8")) > max_line_bytes:
                    preview = preview[:max_line_bytes] + " (line truncated)"
                matches.append((rel, lineno, preview))

    if not matches:
        return "No matches found"

    kept = matches[:max_matches]
    by_file: dict[str, list[tuple[int, str]]] = {}
    for path, lineno, line in kept:
        by_file.setdefault(path, []).append((lineno, line))
    lines_out: list[str] = []
    for path, rows in by_file.items():
        lines_out.append(path)
        for lineno, line in rows:
            lines_out.append(f"  Line {lineno}: {line}")
    header = (
        f"Found {len(kept)} matches"
        if len(kept) == len(matches)
        else f"Found {len(kept)} of {len(matches)} matches"
    )
    body = "\n".join(lines_out)
    if len(kept) < len(matches):
        body += "\n(Full grep result was limited by maxMatches.)"
    return f"{header}\n\n{body}"


def apply(ctx: Context, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
    """Mount glob and grep onto ``ctx.tools`` (dsh's ``tool-fs-search`` row)."""
    cfg = {**(config or {}), **kwargs}
    max_results = int(cfg.get("globMaxResults", DEFAULT_MAX_RESULTS))
    max_matches = int(cfg.get("grepMaxMatches", DEFAULT_MAX_MATCHES))
    max_line_bytes = int(cfg.get("grepMaxLineBytes", DEFAULT_MAX_LINE_BYTES))
    configured_root = cfg.get("workspace_root")

    def _root() -> Path | str | None:
        return configured_root or getattr(ctx, "workspace_dir", None)

    async def _glob(args: dict[str, Any]) -> str:
        return handle_glob(args, workspace_root=_root(), max_results=max_results)

    async def _grep(args: dict[str, Any]) -> str:
        return handle_grep(
            args,
            workspace_root=_root(),
            max_matches=max_matches,
            max_line_bytes=max_line_bytes,
        )

    ctx.tools.register(
        define_tool(
            name="glob",
            description=(
                "Find files, not directories, whose paths match a glob pattern, "
                f"including hidden files. Returns up to {max_results} paths in "
                "modification-time order."
            ),
            parameters={
                "pattern": {
                    "type": "string",
                    "description": (
                        "Glob pattern to match file paths (e.g. '**/*.py'). A "
                        "pattern with no slash matches the basename at any depth."
                    ),
                    "required": True,
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Directory to search in. Defaults to the session workspace; "
                        "a relative path resolves against it."
                    ),
                },
            },
            execute=_glob,
        )
    )
    ctx.tools.register(
        define_tool(
            name="grep",
            description=(
                "Search file contents with a regular expression. Returns matching "
                "lines with line numbers, grouped by file, up to "
                f"{max_matches} matches."
            ),
            parameters={
                "pattern": {
                    "type": "string",
                    "description": "Regular expression to search for.",
                    "required": True,
                },
                "path": {
                    "type": "string",
                    "description": (
                        "File or directory to search. Defaults to the session "
                        "workspace."
                    ),
                },
                "include": {
                    "type": "string",
                    "description": (
                        "One glob filter for which files to search (e.g. '*.py')."
                    ),
                },
            },
            execute=_grep,
        )
    )

    system_prompt = ctx.get("systemPrompt", None, strict=False)
    if system_prompt is not None:
        system_prompt.section(
            {
                "name": "tool:glob",
                "order": system_prompt.get_section_order("TOOL_GLOB"),
                "text": GLOB_SECTION_TEXT,
            }
        )
        system_prompt.section(
            {
                "name": "tool:grep",
                "order": system_prompt.get_section_order("TOOL_GREP"),
                "text": GREP_SECTION_TEXT,
            }
        )


apply.inject = ["tools"]  # type: ignore[attr-defined]

__all__ = [
    "DEFAULT_MAX_LINE_BYTES",
    "DEFAULT_MAX_MATCHES",
    "DEFAULT_MAX_RESULTS",
    "GLOB_SECTION_TEXT",
    "GREP_SECTION_TEXT",
    "apply",
    "handle_glob",
    "handle_grep",
]
