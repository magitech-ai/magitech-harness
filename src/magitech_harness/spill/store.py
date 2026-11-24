"""Kernel-independent storage mechanics for the local spill backend.

Private session-scoped directory selection, safe-name derivation, path-traversal
protection, and the exclusive owner-only write. No animas or harness imports --
this module is pure Python standard library so it can be tested in isolation.

Local filesystem store for spilled text.
"""

from __future__ import annotations

import errno
import hashlib
import os
import tempfile
from typing import Any, TypedDict

DEFAULT_ROOT_PREFIX = "magitech-spill-"


def isErrno(error: Any, code: str) -> bool:
    """Test a caught value for a POSIX system error code."""
    return isinstance(error, OSError) and error.errno == getattr(errno, code, -1)


_default_root: str | None = None


def privateRoot() -> str:
    """Return the lazily-created private per-process spill root."""
    global _default_root
    if _default_root is None:
        _default_root = tempfile.mkdtemp(prefix=DEFAULT_ROOT_PREFIX)
    return _default_root


def encodeSegment(raw: str) -> str:
    """Encode an arbitrary string as one safe path segment.

    Each code unit is kept literal (``[A-Za-z0-9._-]``, minus ``~``) or escaped
    as ``~XXXX``; ``~`` is itself escaped, so the mapping is reversible and
    distinct inputs never collide. The whole-segment tokens ``.``/``..`` are
    escaped so they can never traverse.

    Args:
        raw: Untrusted text (session id / suggested name).

    Returns:
        One injective filesystem-safe path segment.
    """
    if len(raw) == 0:
        return "~"
    if raw == ".":
        return "~002E"
    if raw == "..":
        return "~002E~002E"
    out: list[str] = []
    for ch in raw:
        code = ord(ch)
        if ch != "~" and (chr(code).isalnum() or chr(code) in "._-"):
            out.append(ch)
        else:
            out.append(f"~{code:04X}")
    return "".join(out)


def sessionDir(root: str, sessionId: str) -> str:
    """Derive the stable session-scoped directory under a spill root."""
    hash_hex = hashlib.sha256(sessionId.encode("utf-8")).hexdigest()[:12]
    return os.path.join(root, f"session-{hash_hex}")


class SaveTextOptions(TypedDict):
    """Inputs needed to save a local spill file."""

    root: str
    sessionId: str
    suggestedName: str
    content: str


class SavedText(TypedDict):
    """A written spill file."""

    path: str
    bytes: int


def _write_to_fd(fd: int, data: bytes) -> int:
    """Write bytes to a file descriptor. Exposed for testing."""
    return os.write(fd, data)


def saveTextFile(options: SaveTextOptions) -> SavedText:
    """Write text to a fresh owner-only (0600) file below its session dir.

    Args:
        options: The save request (root, sessionId, suggestedName, content).

    Returns:
        The saved absolute path and UTF-8 byte length.
    """
    directory = sessionDir(options["root"], options["sessionId"])
    data = options["content"].encode("utf-8")
    while True:
        name = f"{os.urandom(6).hex()}-{encodeSegment(options['suggestedName'])}"
        path = os.path.join(directory, name)
        try:
            os.makedirs(directory, mode=0o700, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError:
            # Name collision -- generate a fresh random prefix and retry.
            continue

    _write_to_fd(fd, data)
    os.close(fd)

    return {"path": path, "bytes": len(data)}
