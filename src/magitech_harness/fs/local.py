"""Host-filesystem implementation of FileSystem (ctx.fs)."""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import errno
import os
import re
import shutil
import stat as stat_mod
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, BinaryIO, Literal

from animas import Context

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.fs.patch import PatchHunk, apply_hunks
from magitech_harness.fs.service import FileSystem
from magitech_harness.fs.types import (
    FsDeleteOutcome,
    FsDirEntry,
    FsEditOutcome,
    FsEditRequest,
    FsError,
    FsInfo,
    FsMoveOutcome,
    FsPathInfo,
    FsTarget,
    FsVersion,
    FsWriteIntent,
    FsWriteOutcome,
)

_DEFAULT_DIFF_BASIS_MAX_BYTES = 10 * 1024 * 1024
_MAX_DIFF_BASIS_BYTES = 512 * 1024 * 1024
_BINARY_SAMPLE_BYTES = 8192
_STREAM_CHUNK_BYTES = 65536
_WATCH_POLL_INTERVAL_S = 0.05
_PARENT_SEG_RE = re.compile(r"(?:^|[\\/])\.\.(?:[\\/]|$)")


def _version_after_write(target_key: str) -> FsVersion:
    probed = _probe(target_key)
    return probed[0] if probed is not None else f"missing:{target_key}"


def _read_bytes_bounded(path_str: str, display_path: str, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    with Path(path_str).open("rb") as fh:
        while True:
            chunk = fh.read(_STREAM_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise FsError(
                    f'cannot read "{display_path}": size {total} exceeds {max_bytes}',
                    "FS_TOO_LARGE",
                )
            chunks.append(chunk)
    return b"".join(chunks)


def _watch_snapshot(path_str: str, is_dir: bool) -> Any:
    try:
        if is_dir:
            with os.scandir(path_str) as it:
                entries: list[tuple[str, tuple[int, int, int] | None]] = []
                for entry in sorted(it, key=lambda e: e.name):
                    try:
                        st = entry.stat(follow_symlinks=False)
                        entries.append(
                            (entry.name, (st.st_mtime_ns, st.st_size, st.st_mode))
                        )
                    except OSError:
                        entries.append((entry.name, None))
                return tuple(entries)
        st = os.lstat(path_str)
        return (st.st_mtime_ns, st.st_size, st.st_ino, st.st_mode)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ENOTDIR):
            return None
        raise


def _assert_sandbox_write_allowed(
    sandbox_policy: Any, target: FsTarget, verb: str
) -> None:
    """Enforce the resolved file policy at the filesystem provider boundary."""
    from collections.abc import Mapping

    from magitech_harness.fs.types import FsError

    if sandbox_policy is None:
        return
    if not isinstance(sandbox_policy, Mapping):
        raise FsError("invalid sandbox policy", "FS_SANDBOX_DENIED")
    mode = sandbox_policy.get("mode", "read-only")
    if mode == "danger-full-access":
        return
    raw = str(target.get("targetKey", ""))
    try:
        path = Path(raw).resolve(strict=False)
    except OSError:
        path = Path(raw)
    roots: list[str] = []
    if mode == "workspace-write":
        roots.append(
            str(
                Path(str(sandbox_policy.get("workspaceRoot", ""))).resolve(strict=False)
            )
        )
        for extra in sandbox_policy.get("additionalWritableRoots") or ():
            roots.append(str(Path(str(extra)).resolve(strict=False)))
    roots.append("/dev/null")
    if mode == "workspace-write":
        roots.append("/tmp")
    for root in roots:
        try:
            path.relative_to(root)
            return
        except ValueError:
            continue
    from magitech_harness.fs.types import FsError

    raise FsError(
        f"sandbox denied {verb} to {target.get('displayPath', raw)} under {mode} mode",
        "FS_SANDBOX_DENIED",
    )


def _throw_if_aborted(signal: AbortSignal | None, verb: str) -> None:
    if signal is not None and signal.aborted:
        raise FsError(f"{verb} aborted", "FS_ABORTED")


def expand_home_prefix(path: str) -> str:
    """Expand leading ``~`` or ``~/`` (and ``~\\`` on Windows) against user home."""
    if (
        path == "~"
        or path.startswith("~/")
        or (sys.platform == "win32" and path.startswith("~\\"))
    ):
        return f"{Path.home()}{path[1:]}"
    return path


def _version_of(info: os.stat_result) -> FsVersion:
    return f"{info.st_dev}:{info.st_ino}:{info.st_size}:{info.st_mtime_ns}:{info.st_ctime_ns}"


def _path_type(mode: int) -> Literal["file", "directory", "other"]:
    if stat_mod.S_ISREG(mode):
        return "file"
    if stat_mod.S_ISDIR(mode):
        return "directory"
    return "other"


def _path_link_type(
    mode: int,
) -> Literal["file", "directory", "symlink", "other"]:
    if stat_mod.S_ISLNK(mode):
        return "symlink"
    return _path_type(mode)


def _translate_os_error(verb: str, display_path: str, exc: OSError) -> FsError:
    if exc.errno in (errno.ENOENT, errno.ENOTDIR):
        return FsError(
            f'cannot {verb} "{display_path}": not found',
            "FS_NOT_FOUND",
            cause=exc,
        )
    if exc.errno in (errno.EACCES, errno.EPERM):
        return FsError(
            f'cannot {verb} "{display_path}": permission denied',
            "FS_PERMISSION_DENIED",
            cause=exc,
        )
    if exc.errno == errno.EISDIR:
        return FsError(
            f'cannot {verb} "{display_path}": target is a directory',
            "FS_NOT_REGULAR_FILE",
            cause=exc,
        )
    return FsError(f'cannot {verb} "{display_path}": {exc}', "FS_IO_ERROR", cause=exc)


def _probe(
    path_str: str,
) -> tuple[FsVersion, int, Literal["file", "directory", "other"], int] | None:
    try:
        st = os.stat(path_str)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ENOTDIR):
            return None
        raise
    return (
        _version_of(st),
        stat_mod.S_IMODE(st.st_mode),
        _path_type(st.st_mode),
        st.st_size,
    )


def _probe_nofollow(path_str: str) -> FsPathInfo | None:
    try:
        st = os.lstat(path_str)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ENOTDIR):
            return None
        raise
    return {
        "version": _version_of(st),
        "type": _path_link_type(st.st_mode),
        "size": st.st_size,
    }


def _local_display_path(cwd: str, path: str) -> str:
    abs_cwd = cwd if os.path.isabs(cwd) else os.path.join(os.getcwd(), cwd)
    raw = path if os.path.isabs(path) else f"{abs_cwd}{os.sep}{path}"
    if sys.platform != "win32" and _PARENT_SEG_RE.search(raw):
        return raw
    return os.path.normpath(os.path.join(abs_cwd, path))


def _resolve_local_target(cwd: str, raw_path: str) -> FsTarget:
    path = expand_home_prefix(raw_path)
    if not path.strip():
        raise FsError("file_path must be a non-empty string", "FS_NOT_FOUND")
    display_path = _local_display_path(cwd, path)
    if os.path.exists(display_path):
        try:
            key = os.path.realpath(display_path, strict=True)
            return {"targetKey": key, "displayPath": display_path}
        except OSError as exc:
            raise _translate_os_error("resolve", display_path, exc) from exc

    # Walk ancestor chain to preserve stable key across missing dirs/symlinks
    missing: list[str] = [os.path.basename(display_path)]
    ancestor = os.path.dirname(display_path)
    while True:
        try:
            real_ancestor = os.path.realpath(ancestor, strict=True)
            if ".." in missing:
                raise FsError(
                    f'cannot resolve "{display_path}": parent traversal crosses missing dir',
                    "FS_NOT_FOUND",
                )
            if not os.path.isdir(real_ancestor):
                raise FsError(
                    f'cannot resolve "{display_path}": parent segment is not a directory',
                    "FS_NOT_FOUND",
                )
            return {
                "targetKey": os.path.join(real_ancestor, *missing),
                "displayPath": display_path,
            }
        except OSError as exc:
            if exc.errno == errno.ENOTDIR:
                raise FsError(
                    f'cannot resolve "{display_path}": parent segment is not a directory',
                    "FS_NOT_FOUND",
                    cause=exc,
                ) from exc
            if exc.errno != errno.ENOENT:
                raise _translate_os_error("resolve", display_path, exc) from exc
            parent = os.path.dirname(ancestor)
            if parent == ancestor:
                return {"targetKey": display_path, "displayPath": display_path}
            missing.insert(0, os.path.basename(ancestor))
            ancestor = parent


def _decode_text_bytes(raw: bytes, display_path: str) -> str:
    if b"\x00" in raw[:_BINARY_SAMPLE_BYTES]:
        raise FsError(f'cannot read "{display_path}": binary file', "FS_NOT_TEXT")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FsError(
            f'cannot read "{display_path}": file is not valid UTF-8 text',
            "FS_NOT_TEXT",
            cause=exc,
        ) from exc


def _normalize_line_endings(text: str) -> tuple[str, Literal["crlf", "lf"]]:
    if "\r\n" in text:
        return text.replace("\r\n", "\n"), "crlf"
    return text, "lf"


def _restore_line_endings(text: str, style: Literal["crlf", "lf"]) -> str:
    if style == "crlf":
        return text.replace("\n", "\r\n")
    return text


def _write_atomic(
    path: Path, data: bytes, *, mode: int, create_if_absent: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mth-stage-", dir=str(path.parent)))
    temp_path = staging / f".tmp-{uuid.uuid4().hex}"
    try:
        fd = os.open(
            str(temp_path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if sys.platform != "win32":
            os.chmod(temp_path, mode)
        if create_if_absent:
            os.link(temp_path, path)
        else:
            os.replace(temp_path, path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _open_binary_read(path_str: str) -> BinaryIO:
    return Path(path_str).open("rb")


def _read_binary_range(path_str: str, offset: int, length: int) -> bytes:
    with Path(path_str).open("rb") as handle:
        handle.seek(offset)
        return handle.read(length)


class LocalFileSystem(FileSystem):
    """Host-filesystem provider for ``ctx.fs``."""

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx)
        cfg = config or {}
        if "cwd" in cfg:
            raw_cwd = cfg["cwd"]
            if not isinstance(raw_cwd, str) or not raw_cwd.strip():
                raise ValueError("fs-local: cwd must be a non-empty string")
            self._cwd = raw_cwd
        else:
            self._cwd = os.getcwd()
        if "diff_basis_max_bytes" in cfg or "diffBasisMaxBytes" in cfg:
            raw_limit = cfg.get("diff_basis_max_bytes", cfg.get("diffBasisMaxBytes"))
            if (
                isinstance(raw_limit, bool)
                or not isinstance(raw_limit, int)
                or raw_limit <= 0
                or raw_limit > _MAX_DIFF_BASIS_BYTES
            ):
                raise ValueError(
                    "fs-local: diffBasisMaxBytes must be a positive integer "
                    f"no greater than {_MAX_DIFF_BASIS_BYTES}"
                )
            self._diff_basis_max_bytes = raw_limit
        else:
            self._diff_basis_max_bytes = _DEFAULT_DIFF_BASIS_MAX_BYTES
        self._locks: dict[str, list[Any]] = {}
        self._watchers: set[asyncio.Task[None]] = set()

        async def _dispose() -> None:
            tasks = list(self._watchers)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        ctx.effect(lambda: _dispose, label="fs-local lifecycle")

    @asynccontextmanager
    async def _lock_targets(self, *keys: str) -> AsyncIterator[None]:
        ordered = sorted(set(keys))
        acquired: list[asyncio.Lock] = []
        ref_entries: list[tuple[str, list[Any]]] = []
        try:
            for key in ordered:
                entry = self._locks.get(key)
                if entry is None:
                    entry = [asyncio.Lock(), 0]
                    self._locks[key] = entry
                entry[1] += 1
                ref_entries.append((key, entry))
                await entry[0].acquire()
                acquired.append(entry[0])
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()
            for key, entry in reversed(ref_entries):
                entry[1] -= 1
                if entry[1] == 0:
                    del self._locks[key]

    async def watch(
        self,
        target: FsTarget,
        changed: Callable[[Exception | None], None],
        signal: AbortSignal,
    ) -> Callable[[], Awaitable[None]]:
        """Install polling watch handle until closed or aborted."""
        _throw_if_aborted(signal, "watch")
        target_path = os.path.abspath(self.process_path(target))
        info = await self.stat(target, signal)
        _throw_if_aborted(signal, "watch")
        is_dir = info is not None and info["type"] == "directory"
        prev_snapshot = _watch_snapshot(target_path, is_dir)
        stop_event = asyncio.Event()
        unsub = signal.on_abort(stop_event.set)

        async def _poll() -> None:
            nonlocal prev_snapshot
            try:
                while not stop_event.is_set() and not signal.aborted:
                    try:
                        await asyncio.wait_for(
                            stop_event.wait(), timeout=_WATCH_POLL_INTERVAL_S
                        )
                        break
                    except TimeoutError:
                        pass
                    try:
                        current = _watch_snapshot(target_path, is_dir)
                    except OSError as exc:
                        changed(exc)
                        return
                    if current != prev_snapshot:
                        prev_snapshot = current
                        changed(None)
            finally:
                unsub()

        task = asyncio.create_task(_poll())
        self._watchers.add(task)
        task.add_done_callback(self._watchers.discard)

        async def _close() -> None:
            stop_event.set()
            unsub()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        return _close

    async def resolve(
        self, path: str, opts: Mapping[str, Any] | None = None
    ) -> FsTarget:
        """Resolve *path* relative to ``opts['cwd']`` or configured cwd."""
        signal = opts.get("signal") if opts else None
        _throw_if_aborted(signal, "resolve")
        base_cwd = (
            str(opts["cwd"])
            if opts and isinstance(opts.get("cwd"), str) and opts["cwd"]
            else self._cwd
        )
        return _resolve_local_target(base_cwd, path)

    def process_path(self, target: FsTarget) -> str:
        """Return canonical host path for *target*."""
        return target["targetKey"]

    def process_path_from_host_path(self, host_path: str) -> str | None:
        """Resolve absolute host paths onto LocalFileSystem targets."""
        return os.path.abspath(host_path) if os.path.isabs(host_path) else None

    def file_url(self, target: FsTarget) -> str:
        """Return canonical ``file://`` URI."""
        return Path(target["targetKey"]).as_uri()

    def contains(self, parent: FsTarget, child: FsTarget) -> bool:
        """Return True when *child* equals or lies under *parent*."""
        p_key = os.path.normcase(os.path.normpath(parent["targetKey"]))
        c_key = os.path.normcase(os.path.normpath(child["targetKey"]))
        if p_key == c_key:
            return True
        prefix = p_key if p_key.endswith(os.sep) else p_key + os.sep
        return c_key.startswith(prefix)

    async def stat(
        self, target: FsTarget, signal: AbortSignal | None = None
    ) -> FsInfo | None:
        """Stat target following symlinks; returns None when absent."""
        _throw_if_aborted(signal, "stat")
        probed = _probe(target["targetKey"])
        if probed is None:
            return None
        version, _mode, kind, size = probed
        return {"version": version, "type": kind, "size": size}

    async def lstat(
        self,
        path: str,
        opts: Mapping[str, Any] | None = None,
        signal: AbortSignal | None = None,
    ) -> FsPathInfo | None:
        """Inspect path without following a trailing symlink."""
        _throw_if_aborted(signal, "lstat")
        expanded = expand_home_prefix(path)
        if not expanded.strip():
            raise FsError("file_path must be a non-empty string", "FS_NOT_FOUND")
        base_cwd = (
            str(opts["cwd"])
            if opts and isinstance(opts.get("cwd"), str) and opts["cwd"]
            else self._cwd
        )
        display_path = _local_display_path(base_cwd, expanded)
        return _probe_nofollow(display_path)

    def _require_regular_file(
        self, target: FsTarget, verb: str
    ) -> tuple[FsVersion, int, int]:
        probed = _probe(target["targetKey"])
        if probed is None:
            raise FsError(
                f'cannot {verb} "{target["displayPath"]}": not found',
                "FS_NOT_FOUND",
            )
        version, mode, kind, size = probed
        if kind != "file":
            raise FsError(
                f'cannot {verb} "{target["displayPath"]}": not a regular file',
                "FS_NOT_REGULAR_FILE",
            )
        return version, mode, size

    async def read_text(
        self, target: FsTarget, signal: AbortSignal | None = None
    ) -> str:
        """Read entire regular file as validated UTF-8 text."""
        _throw_if_aborted(signal, "read")
        self._require_regular_file(target, "read")
        try:
            raw = Path(target["targetKey"]).read_bytes()
        except OSError as exc:
            raise _translate_os_error("read", target["displayPath"], exc) from exc
        _throw_if_aborted(signal, "read")
        return _decode_text_bytes(raw, target["displayPath"])

    async def stream_text(
        self, target: FsTarget, signal: AbortSignal | None = None
    ) -> AsyncIterator[str]:
        """Yield validated UTF-8 text chunks for *target*."""
        _throw_if_aborted(signal, "read")
        self._require_regular_file(target, "read")

        async def _chunks() -> AsyncIterator[str]:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
            first_sample = bytearray()
            try:
                with _open_binary_read(target["targetKey"]) as handle:
                    while True:
                        _throw_if_aborted(signal, "read")
                        chunk = handle.read(_STREAM_CHUNK_BYTES)
                        if not chunk:
                            break
                        if len(first_sample) < _BINARY_SAMPLE_BYTES:
                            first_sample.extend(
                                chunk[: _BINARY_SAMPLE_BYTES - len(first_sample)]
                            )
                            if b"\x00" in first_sample:
                                raise FsError(
                                    f'cannot read "{target["displayPath"]}": binary file',
                                    "FS_NOT_TEXT",
                                )
                        text = decoder.decode(chunk, final=False)
                        if text:
                            yield text
                decoder.decode(b"", final=True)
            except UnicodeDecodeError as exc:
                raise FsError(
                    f'cannot read "{target["displayPath"]}": file is not valid UTF-8 text',
                    "FS_NOT_TEXT",
                    cause=exc,
                ) from exc
            except OSError as exc:
                raise _translate_os_error("read", target["displayPath"], exc) from exc

        return _chunks()

    async def read_bytes(
        self,
        target: FsTarget,
        signal: AbortSignal | None,
        max_bytes: int,
    ) -> bytes:
        """Read raw bytes failing FS_TOO_LARGE when file exceeds *max_bytes*."""
        _throw_if_aborted(signal, "read")
        _ver, _mode, size = self._require_regular_file(target, "read")
        if size > max_bytes:
            raise FsError(
                f'cannot read "{target["displayPath"]}": size {size} exceeds {max_bytes}',
                "FS_TOO_LARGE",
            )
        try:
            return _read_bytes_bounded(
                target["targetKey"], target["displayPath"], max_bytes
            )
        except OSError as exc:
            raise _translate_os_error("read", target["displayPath"], exc) from exc

    async def read_byte_range(
        self,
        target: FsTarget,
        byte_range: Mapping[str, int],
        signal: AbortSignal | None = None,
    ) -> bytes:
        """Read raw byte slice ``[offset, offset + length)``."""
        _throw_if_aborted(signal, "read")
        self._require_regular_file(target, "read")
        offset = int(byte_range["offset"])
        length = int(byte_range["length"])
        if offset < 0 or length < 0:
            raise ValueError("offset and length must be non-negative")
        if length == 0:
            return b""
        try:
            return _read_binary_range(target["targetKey"], offset, length)
        except OSError as exc:
            raise _translate_os_error("read", target["displayPath"], exc) from exc

    async def list_dir(
        self, target: FsTarget, signal: AbortSignal | None = None
    ) -> list[FsDirEntry]:
        """List direct children of *target* sorted deterministically by name."""
        _throw_if_aborted(signal, "list")
        probed = _probe(target["targetKey"])
        if probed is None:
            raise FsError(
                f'cannot list "{target["displayPath"]}": not found',
                "FS_NOT_FOUND",
            )
        if probed[2] != "directory":
            raise FsError(
                f'cannot list "{target["displayPath"]}": not a directory',
                "FS_NOT_DIRECTORY",
            )
        try:
            names = sorted(os.listdir(target["targetKey"]))
        except OSError as exc:
            raise _translate_os_error("list", target["displayPath"], exc) from exc
        entries: list[FsDirEntry] = []
        for name in names:
            _throw_if_aborted(signal, "list")
            child_display = os.path.join(target["displayPath"], name)
            child_raw = os.path.join(target["targetKey"], name)
            try:
                child_key = os.path.realpath(child_raw, strict=True)
            except OSError:
                child_key = child_raw
            child_probe = _probe(child_key)
            kind: Literal["file", "directory", "other"] = (
                child_probe[2] if child_probe is not None else "other"
            )
            entry: FsDirEntry = {
                "name": name,
                "type": kind,
                "target": {"targetKey": child_key, "displayPath": child_display},
            }
            if child_probe is not None:
                entry["version"] = child_probe[0]
                if kind == "file":
                    entry["size"] = child_probe[3]
            entries.append(entry)
        return entries

    def _read_diff_basis(self, path_str: str) -> str | None:
        try:
            st = os.stat(path_str)
            if (
                not stat_mod.S_ISREG(st.st_mode)
                or st.st_size >= self._diff_basis_max_bytes
            ):
                return None
            raw = Path(path_str).read_bytes()
        except OSError:
            return None
        if b"\x00" in raw[:_BINARY_SAMPLE_BYTES]:
            return None
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
        normalized, _ = _normalize_line_endings(text)
        return normalized

    async def write_text(
        self,
        target: FsTarget,
        content: str,
        expected: FsWriteIntent | None = None,
        signal: AbortSignal | None = None,
        sandbox_policy: Any = None,
    ) -> FsWriteOutcome:
        """Atomically create or overwrite UTF-8 text."""
        _assert_sandbox_write_allowed(sandbox_policy, target, "write")
        _throw_if_aborted(signal, "write")
        async with self._lock_targets(target["targetKey"]):
            _throw_if_aborted(signal, "write")
            return self._write_text_locked(target, content, expected)

    def _write_text_locked(
        self,
        target: FsTarget,
        content: str,
        expected: FsWriteIntent | None,
        *,
        mode_override: int | None = None,
    ) -> FsWriteOutcome:
        key_path = Path(target["targetKey"])
        probed = _probe(target["targetKey"])
        if expected is not None:
            if expected["kind"] == "createIfAbsent":
                if probed is not None:
                    raise FsError(
                        f'cannot create "{target["displayPath"]}": file already exists',
                        "FS_NOT_OBSERVED",
                    )
            elif probed is None or probed[0] != expected["version"]:
                raise FsError(
                    f'cannot write "{target["displayPath"]}": file modified concurrently',
                    "FS_STALE_VERSION",
                )
        if probed is not None and probed[2] != "file":
            raise FsError(
                f'cannot write "{target["displayPath"]}": not a regular file',
                "FS_NOT_REGULAR_FILE",
            )
        operation: Literal["create", "update"] = (
            "create" if probed is None else "update"
        )
        before = (
            self._read_diff_basis(target["targetKey"]) if probed is not None else None
        )
        mode = (
            mode_override
            if mode_override is not None
            else (probed[1] if probed is not None else 0o644)
        )
        after_lf, _ = _normalize_line_endings(content)
        encoded = content.encode("utf-8")
        create_if_absent = expected is not None and expected["kind"] == "createIfAbsent"
        try:
            _write_atomic(
                key_path,
                encoded,
                mode=mode,
                create_if_absent=create_if_absent,
            )
        except FileExistsError as exc:
            raise FsError(
                f'cannot create "{target["displayPath"]}": file already exists',
                "FS_NOT_OBSERVED",
                cause=exc,
            ) from exc
        except OSError as exc:
            raise _translate_os_error("write", target["displayPath"], exc) from exc
        return {
            "operation": operation,
            "version": _version_after_write(target["targetKey"]),
            "before": before,
            "after": after_lf,
        }

    def _read_editable_file(
        self,
        target: FsTarget,
        verb: str,
        expected: Mapping[str, FsVersion] | None,
    ) -> tuple[str, Literal["crlf", "lf"], int]:
        probed = _probe(target["targetKey"])
        if probed is None:
            raise FsError(
                f'cannot {verb} "{target["displayPath"]}": stale or missing',
                "FS_STALE_VERSION",
            )
        version, mode, kind, _size = probed
        if kind != "file":
            raise FsError(
                f'cannot {verb} "{target["displayPath"]}": not a regular file',
                "FS_NOT_REGULAR_FILE",
            )
        if expected is not None and version != expected["version"]:
            raise FsError(
                f'cannot {verb} "{target["displayPath"]}": stale version',
                "FS_STALE_VERSION",
            )
        try:
            raw = Path(target["targetKey"]).read_bytes()
        except OSError as exc:
            raise _translate_os_error(verb, target["displayPath"], exc) from exc
        decoded = _decode_text_bytes(raw, target["displayPath"])
        normalized, line_style = _normalize_line_endings(decoded)
        return normalized, line_style, mode

    async def edit_text(
        self,
        target: FsTarget,
        edit: FsEditRequest,
        expected: Mapping[str, FsVersion] | None = None,
        signal: AbortSignal | None = None,
        sandbox_policy: Any = None,
    ) -> FsEditOutcome:
        """Atomically perform a literal text replacement."""
        _assert_sandbox_write_allowed(sandbox_policy, target, "write")
        _throw_if_aborted(signal, "edit")
        async with self._lock_targets(target["targetKey"]):
            _throw_if_aborted(signal, "edit")
            before, style, mode = self._read_editable_file(target, "edit", expected)
            old_lf, _ = _normalize_line_endings(edit["oldString"])
            new_lf, _ = _normalize_line_endings(edit["newString"])
            if not old_lf:
                raise FsError("oldString must be non-empty", "FS_EDIT_NOT_FOUND")
            count = before.count(old_lf)
            if count == 0:
                raise FsError(
                    f'cannot edit "{target["displayPath"]}": oldString not found',
                    "FS_EDIT_NOT_FOUND",
                )
            if count > 1 and not edit["replaceAll"]:
                raise FsError(
                    f'cannot edit "{target["displayPath"]}": found {count} matches',
                    "FS_AMBIGUOUS_EDIT",
                )
            after = (
                before.replace(old_lf, new_lf)
                if edit["replaceAll"]
                else before.replace(old_lf, new_lf, 1)
            )
            encoded = _restore_line_endings(after, style).encode("utf-8")
            try:
                _write_atomic(Path(target["targetKey"]), encoded, mode=mode)
            except OSError as exc:
                raise _translate_os_error("edit", target["displayPath"], exc) from exc
            return {
                "version": _version_after_write(target["targetKey"]),
                "before": before,
                "after": after,
            }

    async def patch_text(
        self,
        target: FsTarget,
        hunks: Sequence[PatchHunk],
        expected: Mapping[str, FsVersion] | None = None,
        signal: AbortSignal | None = None,
        sandbox_policy: Any = None,
    ) -> FsEditOutcome:
        """Atomically apply parsed patch hunks."""
        _assert_sandbox_write_allowed(sandbox_policy, target, "write")
        _throw_if_aborted(signal, "patch")
        async with self._lock_targets(target["targetKey"]):
            _throw_if_aborted(signal, "patch")
            before, style, mode = self._read_editable_file(target, "patch", expected)
            applied = apply_hunks(before, hunks)
            if not applied["ok"]:
                failure = applied["failure"]
                raise FsError(
                    f'cannot patch "{target["displayPath"]}": unmatched {failure["stage"]} at line {failure["hunkLineNumber"]}',
                    "FS_PATCH_MISMATCH",
                )
            after = applied["text"]
            encoded = _restore_line_endings(after, style).encode("utf-8")
            try:
                _write_atomic(Path(target["targetKey"]), encoded, mode=mode)
            except OSError as exc:
                raise _translate_os_error("patch", target["displayPath"], exc) from exc
            return {
                "version": _version_after_write(target["targetKey"]),
                "before": before,
                "after": after,
            }

    async def delete_file(
        self,
        target: FsTarget,
        expected: Mapping[str, FsVersion] | None = None,
        signal: AbortSignal | None = None,
        sandbox_policy: Any = None,
    ) -> FsDeleteOutcome:
        """Remove a regular file with optional version check."""
        _assert_sandbox_write_allowed(sandbox_policy, target, "write")
        _throw_if_aborted(signal, "delete")
        async with self._lock_targets(target["targetKey"]):
            _throw_if_aborted(signal, "delete")
            probed = _probe(target["targetKey"])
            if probed is None:
                raise FsError(
                    f'cannot delete "{target["displayPath"]}": not found',
                    "FS_STALE_VERSION",
                )
            if probed[2] != "file":
                raise FsError(
                    f'cannot delete "{target["displayPath"]}": not a regular file',
                    "FS_NOT_REGULAR_FILE",
                )
            if expected is not None and probed[0] != expected["version"]:
                raise FsError(
                    f'cannot delete "{target["displayPath"]}": stale version',
                    "FS_STALE_VERSION",
                )
            before = self._read_diff_basis(target["targetKey"])
            try:
                Path(target["targetKey"]).unlink()
            except OSError as exc:
                raise _translate_os_error("delete", target["displayPath"], exc) from exc
            return {"before": before}

    async def move_file(
        self,
        source: FsTarget,
        destination: FsTarget,
        options: Mapping[str, Any] | None = None,
        signal: AbortSignal | None = None,
        sandbox_policy: Any = None,
    ) -> FsMoveOutcome:
        """Publish source (optionally patched) at destination, then remove source."""
        _assert_sandbox_write_allowed(sandbox_policy, source, "move")
        _assert_sandbox_write_allowed(sandbox_policy, destination, "move")
        _throw_if_aborted(signal, "move")
        if source["targetKey"] == destination["targetKey"]:
            raise FsError(
                f'cannot move "{source["displayPath"]}" onto itself',
                "FS_SAME_TARGET",
            )
        opts = options or {}
        hunks: Sequence[PatchHunk] | None = opts.get("hunks")
        expected_source: Mapping[str, FsVersion] | None = opts.get("expectedSource")
        expected_dest: FsWriteIntent | None = opts.get("expectedDestination")

        async with self._lock_targets(source["targetKey"], destination["targetKey"]):
            _throw_if_aborted(signal, "move")
            before, style, mode = self._read_editable_file(
                source, "move", expected_source
            )
            source_before = self._read_diff_basis(source["targetKey"])
            if hunks is not None:
                applied = apply_hunks(before, hunks)
                if not applied["ok"]:
                    failure = applied["failure"]
                    raise FsError(
                        f'cannot move "{source["displayPath"]}": unmatched {failure["stage"]} at line {failure["hunkLineNumber"]}',
                        "FS_PATCH_MISMATCH",
                    )
                next_lf = applied["text"]
            else:
                next_lf = before
            content = _restore_line_endings(next_lf, style)
            dest_outcome = self._write_text_locked(
                destination,
                content,
                expected_dest,
                mode_override=mode if sys.platform != "win32" else None,
            )
            try:
                Path(source["targetKey"]).unlink()
            except OSError as exc:
                translated = _translate_os_error("delete", source["displayPath"], exc)
                return {
                    "sourceBefore": source_before,
                    "destination": dest_outcome,
                    "sourceDeleted": False,
                    "sourceFailure": {
                        "code": translated.code,
                        "message": str(translated),
                    },
                }
            return {
                "sourceBefore": source_before,
                "destination": dest_outcome,
                "sourceDeleted": True,
            }


__all__ = ["LocalFileSystem", "expand_home_prefix"]
