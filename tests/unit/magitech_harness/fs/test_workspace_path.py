"""Unit tests for workspace path helpers."""

from __future__ import annotations

from magitech_harness.fs.workspace_path import (
    abbreviate_home_path,
    file_address_for,
    is_absolute_workspace_path,
    path_parts_of,
    resolve_workspace_path,
    workspace_title_of,
)


def test_workspace_path_classification_and_resolution() -> None:
    """Classifies and resolves POSIX and Windows paths uniformly."""
    assert is_absolute_workspace_path("/tmp/project")
    assert is_absolute_workspace_path("C:\\repo")
    assert is_absolute_workspace_path("\\\\server\\share\\dir")
    assert not is_absolute_workspace_path("src/app.py")

    assert resolve_workspace_path("/repo/", "./src/main.py") == "/repo/src/main.py"
    assert resolve_workspace_path("C:\\repo\\", ".\\a.py") == "C:\\repo\\a.py"
    assert resolve_workspace_path(None, "a.py") == "a.py"
    assert resolve_workspace_path("/repo", "/etc/hosts") == "/etc/hosts"


def test_workspace_display_and_file_address() -> None:
    """Abbreviates home paths, splits parts, and builds resource addresses."""
    assert abbreviate_home_path("/home/alice/proj", "/home/alice") == "~/proj"
    assert abbreviate_home_path("/home/alice", "/home/alice/") == "~"
    assert abbreviate_home_path("/opt/x", "/home/alice") == "/opt/x"
    assert abbreviate_home_path("/home/alice", "") == "/home/alice"
    assert abbreviate_home_path("/opt/x", "/") == "/opt/x"
    assert abbreviate_home_path("C:\\Users\\a", "/home/a") == "C:\\Users\\a"

    assert workspace_title_of("/a/b/repo/") == "repo"
    assert workspace_title_of("C:\\work\\proj\\") == "proj"
    assert path_parts_of("/a/b/c.txt") == {"directory": "/a/b/", "name": "c.txt"}
    assert path_parts_of("/") == {"directory": "", "name": "/"}

    assert (
        file_address_for("s1", "/repo", "/repo/src/a.py")
        == "mth-resource://file/s1/src/a.py"
    )
    assert file_address_for("s1", "/repo", "/repo") == "mth-resource://file/s1/"
    assert (
        file_address_for("s1", "/repo", "/etc/passwd")
        == "mth-resource://file/s1//etc/passwd"
    )
    assert file_address_for("s1", None, "rel/a.py") == "mth-resource://file/s1/rel/a.py"
