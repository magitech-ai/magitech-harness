"""Unit tests for FileSystem and LocalFileSystem."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from animas import Context

from magitech_harness.agent.cancellation import AbortController
from magitech_harness.fs.local import LocalFileSystem
from magitech_harness.fs.patch import PatchHunk
from magitech_harness.fs.service import FileSystem
from magitech_harness.fs.types import FsError


@pytest.mark.asyncio
async def test_abstract_filesystem_guard_and_defaults() -> None:
    """Rejects direct instantiation of abstract FileSystem."""
    ctx = Context()
    with pytest.raises(TypeError, match="abstract"):
        FileSystem(ctx)

    class _MinimalFs(FileSystem):
        pass

    assert _MinimalFs.__abstractmethods__


@pytest.mark.asyncio
async def test_local_filesystem_resolve_stat_lstat_containment_and_uris(
    tmp_path: Path,
) -> None:
    """Resolves symlinks and missing ancestors, probes metadata, and checks containment."""
    ctx = Context()
    ctx.use(LocalFileSystem, {"cwd": str(tmp_path)})
    await ctx.flush()
    fs: FileSystem = ctx.fs

    assert fs.sandbox_mode is None
    assert fs.sandboxMode is None
    assert fs.process_path_from_host_path(str(tmp_path / "x")) == str(tmp_path / "x")

    real_file = tmp_path / "real.txt"
    real_file.write_text("hello", encoding="utf-8")
    link_file = tmp_path / "alias.txt"
    link_file.symlink_to(real_file)

    t_real = await fs.resolve("real.txt")
    t_alias = await fs.resolve(str(link_file))
    assert t_real["targetKey"] == t_alias["targetKey"]
    assert t_alias["displayPath"] == str(link_file)
    assert fs.process_path(t_real) == t_real["targetKey"]
    assert fs.file_url(t_real).startswith("file://")

    root_target = await fs.resolve(str(tmp_path))
    assert fs.contains(root_target, t_real) is True
    outside = await fs.resolve(str(tmp_path.parent))
    assert fs.contains(root_target, outside) is False

    info = await fs.stat(t_real)
    assert info is not None
    assert info["type"] == "file"
    assert info["size"] == 5

    l_alias = await fs.lstat("alias.txt")
    assert l_alias is not None
    assert l_alias["type"] == "symlink"
    assert await fs.lstat("does-not-exist.txt") is None

    # Missing file preserves stable key across missing nested directory creation
    missing = await fs.resolve("nested/sub/new.txt")
    assert await fs.stat(missing) is None
    assert fs.contains(root_target, missing) is True

    # Invalid empty path or traversing through a regular file fails FS_NOT_FOUND
    with pytest.raises(FsError) as empty_err:
        await fs.resolve("   ")
    assert empty_err.value.code == "FS_NOT_FOUND"

    with pytest.raises(FsError) as seg_err:
        await fs.resolve("real.txt/child.txt")
    assert seg_err.value.code == "FS_NOT_FOUND"


@pytest.mark.asyncio
async def test_local_filesystem_read_stream_range_and_listdir(tmp_path: Path) -> None:
    """Reads text/bytes/ranges, rejects binaries and non-files, lists dirs."""
    ctx = Context()
    ctx.use(LocalFileSystem, {"cwd": str(tmp_path)})
    await ctx.flush()
    fs = ctx.fs

    txt = tmp_path / "sample.txt"
    txt.write_text("line1\nline2\n", encoding="utf-8", newline="\n")
    bin_file = tmp_path / "bin.dat"
    bin_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x01\x02")
    sub = tmp_path / "b_dir"
    sub.mkdir()
    (sub / "z.txt").write_text("z", encoding="utf-8")
    (sub / "a.txt").write_text("a", encoding="utf-8")

    t_txt = await fs.resolve("sample.txt")
    assert await fs.read_text(t_txt) == "line1\nline2\n"
    chunks = [chunk async for chunk in await fs.stream_text(t_txt)]
    assert "".join(chunks) == "line1\nline2\n"
    assert await fs.read_bytes(t_txt, None, 100) == b"line1\nline2\n"
    assert await fs.read_byte_range(t_txt, {"offset": 2, "length": 5}) == b"ne1\nl"

    with pytest.raises(FsError) as too_large:
        await fs.read_bytes(t_txt, None, 4)
    assert too_large.value.code == "FS_TOO_LARGE"

    t_bin = await fs.resolve("bin.dat")
    with pytest.raises(FsError) as not_text:
        await fs.read_text(t_bin)
    assert not_text.value.code == "FS_NOT_TEXT"

    with pytest.raises(FsError) as stream_not_text:
        async for _chunk in await fs.stream_text(t_bin):
            pass  # pragma: no cover
    assert stream_not_text.value.code == "FS_NOT_TEXT"

    t_sub = await fs.resolve("b_dir")
    entries = await fs.list_dir(t_sub)
    assert [e["name"] for e in entries] == ["a.txt", "z.txt"]
    with pytest.raises(FsError) as not_reg:
        await fs.read_text(t_sub)
    assert not_reg.value.code == "FS_NOT_REGULAR_FILE"
    with pytest.raises(FsError) as not_dir:
        await fs.list_dir(t_txt)
    assert not_dir.value.code == "FS_NOT_DIRECTORY"

    # Aborted read fails FS_ABORTED
    ctrl = AbortController()
    ctrl.abort()
    with pytest.raises(FsError) as aborted:
        await fs.read_text(t_txt, ctrl.signal)
    assert aborted.value.code == "FS_ABORTED"


@pytest.mark.asyncio
async def test_local_filesystem_write_edit_patch_delete_move(tmp_path: Path) -> None:
    """Executes guarded atomic mutations and reports stale/mismatch/same-target errors."""
    ctx = Context()
    ctx.use(LocalFileSystem, {"cwd": str(tmp_path)})
    await ctx.flush()
    fs = ctx.fs

    target = await fs.resolve("nested/doc.txt")
    # Guarded create
    w1 = await fs.write_text(target, "alpha\nbeta\n", {"kind": "createIfAbsent"})
    assert w1["operation"] == "create"
    assert w1["before"] is None
    assert w1["after"] == "alpha\nbeta\n"

    # Repeat createIfAbsent fails FS_NOT_OBSERVED
    with pytest.raises(FsError) as exists_err:
        await fs.write_text(target, "x", {"kind": "createIfAbsent"})
    assert exists_err.value.code == "FS_NOT_OBSERVED"

    # Edit with stale version fails FS_STALE_VERSION
    with pytest.raises(FsError) as stale_err:
        await fs.edit_text(
            target,
            {"oldString": "alpha", "newString": "ALPHA", "replaceAll": False},
            {"version": "0:0:0:0:0"},
        )
    assert stale_err.value.code == "FS_STALE_VERSION"

    # Valid guarded edit (CRLF preservation check too)
    crlf_target = await fs.resolve("crlf.txt")
    w_crlf = await fs.write_text(crlf_target, "one\r\ntwo\r\n")
    e1 = await fs.edit_text(
        crlf_target,
        {"oldString": "one", "newString": "ONE", "replaceAll": False},
        {"version": w_crlf["version"]},
    )
    assert e1["before"] == "one\ntwo\n"
    assert e1["after"] == "ONE\ntwo\n"
    assert (tmp_path / "crlf.txt").read_bytes() == b"ONE\r\ntwo\r\n"

    # Ambiguous and missing edits
    dup_target = await fs.resolve("dup.txt")
    await fs.write_text(dup_target, "x\nx\n")
    with pytest.raises(FsError) as ambig:
        await fs.edit_text(
            dup_target,
            {"oldString": "x", "newString": "y", "replaceAll": False},
        )
    assert ambig.value.code == "FS_AMBIGUOUS_EDIT"
    e_all = await fs.edit_text(
        dup_target,
        {"oldString": "x", "newString": "y", "replaceAll": True},
    )
    assert e_all["after"] == "y\ny\n"
    with pytest.raises(FsError) as not_found_edit:
        await fs.edit_text(
            dup_target,
            {"oldString": "nope", "newString": "y", "replaceAll": False},
        )
    assert not_found_edit.value.code == "FS_EDIT_NOT_FOUND"

    # Patch text + mismatch error
    hunks: list[PatchHunk] = [
        {
            "lineNumber": 1,
            "oldLines": ["alpha"],
            "newLines": ["omega"],
            "isEndOfFile": False,
        }
    ]
    p1 = await fs.patch_text(target, hunks, {"version": w1["version"]})
    assert p1["after"] == "omega\nbeta\n"
    with pytest.raises(FsError) as patch_err:
        await fs.patch_text(
            target,
            [
                {
                    "lineNumber": 1,
                    "oldLines": ["missing"],
                    "newLines": ["x"],
                    "isEndOfFile": False,
                }
            ],
        )
    assert patch_err.value.code == "FS_PATCH_MISMATCH"

    # Move file + same-target check
    with pytest.raises(FsError) as same_err:
        await fs.move_file(target, target, {})
    assert same_err.value.code == "FS_SAME_TARGET"

    dest = await fs.resolve("moved/doc.txt")
    moved = await fs.move_file(
        target,
        dest,
        {
            "hunks": [
                {
                    "lineNumber": 1,
                    "oldLines": ["omega"],
                    "newLines": ["final"],
                    "isEndOfFile": False,
                }
            ],
            "expectedSource": {"version": p1["version"]},
            "expectedDestination": {"kind": "createIfAbsent"},
        },
    )
    assert moved["sourceDeleted"] is True
    assert moved["destination"]["after"] == "final\nbeta\n"
    assert await fs.stat(target) is None

    # Delete file
    del_res = await fs.delete_file(dest, {"version": moved["destination"]["version"]})
    assert del_res["before"] == "final\nbeta\n"
    assert await fs.stat(dest) is None

    # Watch fallback/implementation
    changes: list[Any] = []
    w_ctrl = AbortController()
    unwatch = await fs.watch(dest, lambda err=None: changes.append(err), w_ctrl.signal)
    await unwatch()


@pytest.mark.asyncio
async def test_filesystem_camelcase_aliases_and_watch_default(tmp_path: Path) -> None:
    """Covers camelCase protocol aliases and base FileSystem.watch defaults."""
    ctx = Context()
    ctx.use(LocalFileSystem, {"cwd": str(tmp_path)})
    await ctx.flush()
    fs: FileSystem = ctx.fs

    f1 = await fs.resolve("alias_api.txt")
    await fs.writeText(f1, "hello world\n")
    assert fs.processPath(f1) == fs.process_path(f1)
    assert fs.processPathFromHostPath(str(tmp_path)) == fs.process_path_from_host_path(
        str(tmp_path)
    )
    assert fs.fileUrl(f1) == fs.file_url(f1)
    assert await fs.readText(f1) == "hello world\n"
    assert "".join([c async for c in await fs.streamText(f1)]) == "hello world\n"
    assert await fs.readBytes(f1, None, 100) == b"hello world\n"
    assert await fs.readByteRange(f1, {"offset": 0, "length": 5}) == b"hello"
    e = await fs.editText(
        f1, {"oldString": "world", "newString": "there", "replaceAll": False}
    )
    assert e["after"] == "hello there\n"
    p = await fs.patchText(
        f1,
        [
            {
                "lineNumber": 1,
                "oldLines": ["hello there"],
                "newLines": ["hi there"],
                "isEndOfFile": False,
            }
        ],
    )
    assert p["after"] == "hi there\n"
    d = await fs.resolve(".")
    listed = await fs.listDir(d)
    assert any(item["name"] == "alias_api.txt" for item in listed)
    f2 = await fs.resolve("alias_moved.txt")
    mv = await fs.moveFile(f1, f2, {})
    assert mv["sourceDeleted"] is True
    dl = await fs.deleteFile(f2)
    assert dl["before"] == "hi there\n"

    # Base FileSystem.watch and process_path_from_host_path default behavior
    class _BaseStub(FileSystem):
        async def resolve(self, path: str, opts: Any = None) -> Any:
            return {"targetKey": path, "displayPath": path}

        def process_path(self, target: Any) -> str:
            return target["targetKey"]

        def file_url(self, target: Any) -> str:
            return f"file://{target['targetKey']}"

        def contains(self, parent: Any, child: Any) -> bool:
            return False

        async def stat(self, target: Any, signal: Any = None) -> Any:
            return None

        async def lstat(self, path: str, opts: Any = None, signal: Any = None) -> Any:
            return None

        async def read_text(self, target: Any, signal: Any = None) -> str:
            return ""

        async def stream_text(self, target: Any, signal: Any = None) -> Any:
            async def _gen() -> Any:
                yield ""

            return _gen()

        async def read_bytes(self, target: Any, signal: Any, max_bytes: int) -> bytes:
            return b""

        async def read_byte_range(
            self, target: Any, byte_range: Any, signal: Any = None
        ) -> bytes:
            return b""

        async def list_dir(self, target: Any, signal: Any = None) -> list[Any]:
            return []

        async def write_text(
            self,
            target: Any,
            content: str,
            expected: Any = None,
            signal: Any = None,
            sandbox_policy: Any = None,
        ) -> Any:
            return {}

        async def edit_text(
            self,
            target: Any,
            edit: Any,
            expected: Any = None,
            signal: Any = None,
            sandbox_policy: Any = None,
        ) -> Any:
            return {}

        async def patch_text(
            self,
            target: Any,
            hunks: Any,
            expected: Any = None,
            signal: Any = None,
            sandbox_policy: Any = None,
        ) -> Any:
            return {}

        async def delete_file(
            self,
            target: Any,
            expected: Any = None,
            signal: Any = None,
            sandbox_policy: Any = None,
        ) -> Any:
            return {}

        async def move_file(
            self,
            source: Any,
            destination: Any,
            options: Any = None,
            signal: Any = None,
            sandbox_policy: Any = None,
        ) -> Any:
            return {}

    ctx2 = Context()
    stub = _BaseStub(ctx2)
    assert stub.process_path_from_host_path("/tmp/x") is None
    with pytest.raises(FsError) as watch_io:
        await stub.watch(
            {"targetKey": "/x", "displayPath": "/x"},
            lambda _e=None: None,
            AbortController().signal,
        )
    assert watch_io.value.code == "FS_IO_ERROR"
    aborted = AbortController()
    aborted.abort()
    with pytest.raises(FsError) as watch_aborted:
        await stub.watch(
            {"targetKey": "/x", "displayPath": "/x"},
            lambda _e=None: None,
            aborted.signal,
        )
    assert watch_aborted.value.code == "FS_ABORTED"


@pytest.mark.asyncio
async def test_local_filesystem_error_branches_and_edge_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Covers multi-chunk streaming, encoding failures, special files, and OS errors."""
    import errno
    import stat as stat_mod

    import magitech_harness.fs.local as local_mod

    ctx = Context()
    ctx.use(LocalFileSystem, {"cwd": str(tmp_path), "diff_basis_max_bytes": 16})
    await ctx.flush()
    fs: LocalFileSystem = ctx.fs

    # Home prefix expansion (POSIX + win32) & containment equal target
    monkeypatch.setattr(local_mod.sys, "platform", "linux")
    assert local_mod.expand_home_prefix("~/sub") == f"{Path.home()}/sub"
    if os.name == "nt":
        # Windows' native realpath collapses "missing_dir/.." lexically before I/O;
        # simulate POSIX realpath ENOENT on nonexistent segments to exercise the guard.
        orig_realpath = local_mod.os.path.realpath

        def _posix_like_realpath(p: str, *, strict: bool = False) -> str:
            if strict and "missing_dir" in p:
                raise FileNotFoundError(errno.ENOENT, "missing", p)
            return orig_realpath(p, strict=strict)

        monkeypatch.setattr(local_mod.os.path, "realpath", _posix_like_realpath)
    with pytest.raises(FsError) as missing_dotdot:
        await fs.resolve("missing_dir/../x.txt")
    assert missing_dotdot.value.code == "FS_NOT_FOUND"
    await fs.write_text(await fs.resolve("posix_chmod.txt"), "ok")
    monkeypatch.undo()

    monkeypatch.setattr(local_mod.sys, "platform", "win32")
    assert local_mod.expand_home_prefix("~\\sub") == f"{Path.home()}\\sub"
    win_target = await fs.resolve("sub\\leaf.txt")
    assert win_target["displayPath"] == str(tmp_path / "sub\\leaf.txt")
    await fs.write_text(win_target, "win content")
    norm_dotdot = await fs.resolve("missing_dir/../x.txt")
    assert norm_dotdot["displayPath"].endswith("x.txt")
    # Exercise win32 move_file branch too (mode_override skipped)
    moved_win = await fs.resolve("sub\\moved_leaf.txt")
    await fs.move_file(win_target, moved_win, {})
    monkeypatch.undo()

    root_t = await fs.resolve(".")
    assert fs.contains(root_t, root_t) is True

    # FIFO / special file ("other") and broken symlink in list_dir
    assert local_mod._path_type(stat_mod.S_IFIFO) == "other"
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "pipe_node"
        os.mkfifo(fifo)
    broken_link = tmp_path / "broken_link"
    broken_link.symlink_to(tmp_path / "nonexistent_target")
    listed = await fs.list_dir(root_t)
    by_name = {entry["name"]: entry for entry in listed}
    if hasattr(os, "mkfifo"):
        assert by_name["pipe_node"]["type"] == "other"
    assert by_name["broken_link"]["type"] == "other"

    # Invalid UTF-8 & multi-chunk stream (>64KB) & trailing incomplete UTF-8 sequence
    bad_utf8 = tmp_path / "bad_utf8.txt"
    bad_utf8.write_bytes(b"\xff\xfehello")
    t_bad = await fs.resolve("bad_utf8.txt")
    with pytest.raises(FsError) as bad_read:
        await fs.read_text(t_bad)
    assert bad_read.value.code == "FS_NOT_TEXT"
    with pytest.raises(FsError) as bad_stream:
        async for _c in await fs.stream_text(t_bad):
            pass
    assert bad_stream.value.code == "FS_NOT_TEXT"

    big_txt = tmp_path / "big.txt"
    big_txt.write_text("A" * 70_000, encoding="utf-8")
    t_big = await fs.resolve("big.txt")
    streamed = "".join([c async for c in await fs.stream_text(t_big)])
    assert len(streamed) == 70_000

    # Incomplete trailing multi-byte sequence decoded at final=True + split multi-byte boundary
    trunc_utf8 = tmp_path / "trunc.txt"
    trunc_utf8.write_bytes(b"hello\xc3")
    t_trunc = await fs.resolve("trunc.txt")
    with pytest.raises(FsError) as trunc_err:
        async for _c in await fs.stream_text(t_trunc):
            pass
    assert trunc_err.value.code == "FS_NOT_TEXT"

    # Stream terminating in buffered pending multi-byte tail or empty intermediate chunk
    utf8_edge = tmp_path / "utf8_edge.txt"
    utf8_edge.write_bytes(b"\xc3\xa9")
    t_edge = await fs.resolve("utf8_edge.txt")
    monkeypatch.setattr(local_mod, "_STREAM_CHUNK_BYTES", 1)
    assert "".join([c async for c in await fs.stream_text(t_edge)]) == "\u00e9"
    monkeypatch.undo()

    # _diff_basis skips oversized, OSError, binary, and invalid utf-8 files
    assert await fs.lstat("bad_utf8.txt") is not None
    small_bin = tmp_path / "small_bin.dat"
    small_bin.write_bytes(b"\x00\x01\x02")
    t_small_bin = await fs.resolve("small_bin.dat")
    assert (await fs.write_text(t_small_bin, "text"))["before"] is None
    w_rep = await fs.stat(t_small_bin)
    assert w_rep is not None
    w_ok_rep = await fs.write_text(
        t_small_bin, "text2", {"kind": "replaceIfVersion", "version": w_rep["version"]}
    )
    assert w_ok_rep["after"] == "text2"
    with pytest.raises(FsError) as stale_rep:
        await fs.write_text(
            t_small_bin, "x", {"kind": "replaceIfVersion", "version": "0:0:0:0:0"}
        )
    assert stale_rep.value.code == "FS_STALE_VERSION"
    with pytest.raises(FsError) as stale_del_ver:
        await fs.delete_file(t_small_bin, {"version": "0:0:0:0:0"})
    assert stale_del_ver.value.code == "FS_STALE_VERSION"

    big_out = await fs.write_text(t_bad, "12345678901234567890")
    assert big_out["before"] is None
    assert big_out["after"] == "12345678901234567890"
    oversize_again = await fs.write_text(t_big, "X" * 70_000)
    assert oversize_again["before"] is None

    # Empty lstat & negative/zero byte ranges & missing stats
    with pytest.raises(FsError) as empty_lstat:
        await fs.lstat("   ")
    assert empty_lstat.value.code == "FS_NOT_FOUND"
    assert await fs.read_byte_range(t_bad, {"offset": 0, "length": 0}) == b""
    with pytest.raises(ValueError, match="non-negative"):
        await fs.read_byte_range(t_bad, {"offset": -1, "length": 1})

    # Missing file error paths for read/list_dir (FS_NOT_FOUND) and edit (FS_STALE_VERSION)
    t_absent = await fs.resolve("never_created.txt")
    for fn in (
        lambda: fs.read_text(t_absent),
        lambda: fs.list_dir(t_absent),
    ):
        with pytest.raises(FsError) as not_found_e:
            await fn()
        assert not_found_e.value.code == "FS_NOT_FOUND"
    with pytest.raises(FsError) as edit_missing_e:
        await fs.edit_text(
            t_absent, {"oldString": "a", "newString": "b", "replaceAll": False}
        )
    assert edit_missing_e.value.code == "FS_STALE_VERSION"
    with pytest.raises(FsError) as stale_edit:
        await fs.edit_text(
            t_absent,
            {"oldString": "a", "newString": "b", "replaceAll": False},
            {"version": "0:0:0:0:0"},
        )
    assert stale_edit.value.code == "FS_STALE_VERSION"
    with pytest.raises(FsError) as empty_old:
        await fs.edit_text(
            t_bad, {"oldString": "", "newString": "b", "replaceAll": False}
        )
    assert empty_old.value.code == "FS_EDIT_NOT_FOUND"

    # Non-regular file write/edit/delete
    dir_t = await fs.resolve("dir_here")
    (tmp_path / "dir_here").mkdir()
    for op in (
        lambda: fs.write_text(dir_t, "x"),
        lambda: fs.edit_text(
            dir_t, {"oldString": "x", "newString": "y", "replaceAll": False}
        ),
        lambda: fs.delete_file(dir_t),
    ):
        with pytest.raises(FsError) as non_reg:
            await op()
        assert non_reg.value.code == "FS_NOT_REGULAR_FILE"

    with pytest.raises(FsError) as stale_del_absent:
        await fs.delete_file(t_absent)
    assert stale_del_absent.value.code == "FS_STALE_VERSION"

    # move_file: guarded stale version + patch mismatch + source unlink OS error
    t_src = await fs.resolve("src_move.txt")
    await fs.write_text(t_src, "hello\n")
    t_dst = await fs.resolve("dst_move.txt")
    with pytest.raises(FsError) as mv_patch_err:
        await fs.move_file(
            t_src,
            t_dst,
            {
                "hunks": [
                    {
                        "lineNumber": 1,
                        "oldLines": ["nope"],
                        "newLines": ["x"],
                        "isEndOfFile": False,
                    }
                ]
            },
        )
    assert mv_patch_err.value.code == "FS_PATCH_MISMATCH"
    # Clean dst_move created? patch mismatched before write so not created; move cleanly without hunks while unlink fails
    orig_unlink = Path.unlink

    def _fail_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self.name == "src_move.txt":
            raise PermissionError(errno.EACCES, "denied")
        orig_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _fail_unlink)
    mv_partial = await fs.move_file(t_src, t_dst, {})
    assert mv_partial["sourceDeleted"] is False
    assert mv_partial["sourceFailure"]["code"] == "FS_PERMISSION_DENIED"
    monkeypatch.undo()

    # Translate OS errors directly & injected OS failures for remaining branches
    assert (
        local_mod._translate_os_error(
            "read", "p", FileNotFoundError(errno.ENOENT, "no")
        ).code
        == "FS_NOT_FOUND"
    )
    assert (
        local_mod._translate_os_error(
            "read", "p", PermissionError(errno.EPERM, "perm")
        ).code
        == "FS_PERMISSION_DENIED"
    )
    assert (
        local_mod._translate_os_error(
            "read", "p", IsADirectoryError(errno.EISDIR, "dir")
        ).code
        == "FS_NOT_REGULAR_FILE"
    )
    assert (
        local_mod._translate_os_error("read", "p", OSError(errno.EIO, "io")).code
        == "FS_IO_ERROR"
    )

    # Root-loop / OS error branches in resolve, _probe, read_bytes race, edit/patch/write/delete/listdir OS errors
    real_stat = local_mod.os.stat
    real_lstat = local_mod.os.lstat
    with pytest.raises(OSError):
        monkeypatch.setattr(
            local_mod.os,
            "stat",
            lambda *_a, **_kw: (_ for _ in ()).throw(OSError(errno.EIO, "bad")),
        )
        local_mod._probe("x")
    monkeypatch.setattr(local_mod.os, "stat", real_stat)
    with pytest.raises(OSError):
        monkeypatch.setattr(
            local_mod.os,
            "lstat",
            lambda *_a, **_kw: (_ for _ in ()).throw(OSError(errno.EIO, "bad")),
        )
        local_mod._probe_nofollow("x")
    monkeypatch.setattr(local_mod.os, "lstat", real_lstat)

    # read_bytes size growth race (small file on disk, larger bytes returned)
    import io

    monkeypatch.setattr(
        local_mod.Path,
        "open",
        lambda _self, *_a, **_kw: io.BytesIO(b"12345678901234567890"),
    )
    with pytest.raises(FsError) as race_large:
        await fs.read_bytes(t_edge, None, 10)
    assert race_large.value.code == "FS_TOO_LARGE"
    monkeypatch.undo()

    # _read_diff_basis swallows OSError safely
    monkeypatch.setattr(
        local_mod.Path,
        "read_bytes",
        lambda _self: (_ for _ in ()).throw(OSError(errno.EIO, "io")),
    )
    assert fs._read_diff_basis(t_edge["targetKey"]) is None
    monkeypatch.undo()

    # read_text / stream_text / read_byte_range / list_dir / _read_editable_file / write / edit / patch / delete OSErrors
    monkeypatch.setattr(
        local_mod.Path,
        "read_bytes",
        lambda _self: (_ for _ in ()).throw(PermissionError(errno.EACCES, "denied")),
    )
    monkeypatch.setattr(
        local_mod.Path,
        "open",
        lambda _self, *_a, **_kw: (_ for _ in ()).throw(
            PermissionError(errno.EACCES, "denied")
        ),
    )
    for call in (
        lambda: fs.read_text(t_bad),
        lambda: fs.read_bytes(t_bad, None, 100),
        lambda: fs.edit_text(
            t_bad, {"oldString": "a", "newString": "b", "replaceAll": False}
        ),
    ):
        with pytest.raises(FsError) as perm_e:
            await call()
        assert perm_e.value.code == "FS_PERMISSION_DENIED"
    monkeypatch.undo()

    orig_open = local_mod.Path.open

    def _bad_open(self_path: Path, *args: Any, **kwargs: Any) -> Any:
        if str(self_path).endswith("bad_utf8.txt"):
            raise PermissionError(errno.EACCES, "denied")
        return orig_open(self_path, *args, **kwargs)

    monkeypatch.setattr(local_mod.Path, "open", _bad_open)
    with pytest.raises(FsError) as open_stream_err:
        async for _c in await fs.stream_text(t_bad):
            pass
    assert open_stream_err.value.code == "FS_PERMISSION_DENIED"
    with pytest.raises(FsError) as open_range_err:
        await fs.read_byte_range(t_bad, {"offset": 0, "length": 1})
    assert open_range_err.value.code == "FS_PERMISSION_DENIED"
    monkeypatch.undo()

    monkeypatch.setattr(
        local_mod.os,
        "listdir",
        lambda _p: (_ for _ in ()).throw(PermissionError(errno.EACCES, "denied")),
    )
    with pytest.raises(FsError) as ld_err:
        await fs.list_dir(root_t)
    assert ld_err.value.code == "FS_PERMISSION_DENIED"
    monkeypatch.undo()

    # list_dir realpath fallback & _write_atomic error
    monkeypatch.setattr(
        local_mod,
        "_write_atomic",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            PermissionError(errno.EACCES, "denied")
        ),
    )
    for w_call in (
        lambda: fs.write_text(t_bad, "x"),
        lambda: fs.edit_text(
            t_src, {"oldString": "hello", "newString": "world", "replaceAll": False}
        ),
        lambda: fs.patch_text(
            t_src,
            [
                {
                    "lineNumber": 1,
                    "oldLines": ["hello"],
                    "newLines": ["world"],
                    "isEndOfFile": False,
                }
            ],
        ),
    ):
        with pytest.raises(FsError) as wa_err:
            await w_call()
        assert wa_err.value.code == "FS_PERMISSION_DENIED"
    monkeypatch.undo()

    monkeypatch.setattr(
        Path,
        "unlink",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            PermissionError(errno.EACCES, "denied")
        ),
    )
    with pytest.raises(FsError) as del_perm:
        await fs.delete_file(t_src)
    assert del_perm.value.code == "FS_PERMISSION_DENIED"
    monkeypatch.undo()

    # realpath errors in resolve (strict=True failing) & ENOTDIR during missing-ancestor walk
    monkeypatch.setattr(
        local_mod.os.path,
        "realpath",
        lambda *_a, **_kw: (_ for _ in ()).throw(OSError(errno.EACCES, "denied")),
    )
    with pytest.raises(FsError) as rp_existing:
        await fs.resolve("bad_utf8.txt")
    assert rp_existing.value.code == "FS_PERMISSION_DENIED"
    with pytest.raises(FsError) as rp_missing:
        await fs.resolve("no_such_a/no_such_b.txt")
    assert rp_missing.value.code == "FS_PERMISSION_DENIED"
    monkeypatch.undo()

    # ENOTDIR + root fallback (parent == ancestor) + list_dir child realpath OSError fallback
    real_realpath = local_mod.os.path.realpath

    def _enotdir_realpath(p: str, *, strict: bool = False) -> str:
        if "enotdir_probe" in p:
            raise OSError(errno.ENOTDIR, "not a dir")
        if strict and os.path.dirname(p) == p:
            raise FileNotFoundError(errno.ENOENT, "missing root")
        return real_realpath(p, strict=strict)

    monkeypatch.setattr(local_mod.os.path, "realpath", _enotdir_realpath)
    with pytest.raises(FsError) as enotdir_err:
        await fs.resolve("enotdir_probe/leaf.txt")
    assert enotdir_err.value.code == "FS_NOT_FOUND"
    root_missing = await fs.resolve("/enotdir_other_missing")
    assert root_missing["targetKey"] == f"{tmp_path.drive}{os.sep}enotdir_other_missing"

    def _child_rp_err(p: str, *, strict: bool = False) -> str:
        if "src_move.txt" in p and strict:
            raise OSError(errno.EACCES, "denied")
        return real_realpath(p, strict=strict)

    monkeypatch.setattr(local_mod.os.path, "realpath", _child_rp_err)
    listed2 = await fs.list_dir(root_t)
    assert any(item["name"] == "src_move.txt" for item in listed2)
    monkeypatch.undo()

    # Verify _diff_basis skips symlinks/non-regular files cleanly
    link_t = tmp_path / "dangling_or_dir_link"
    link_t.symlink_to(tmp_path / "dir_here")
    assert fs._read_diff_basis(str(link_t)) is None


@pytest.mark.asyncio
async def test_local_filesystem_fail_loud_watch_link_race_locks_and_host_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Covers fail-loud config, polling watch, createIfAbsent FileExistsError, lock eviction, and host paths."""
    import errno

    import magitech_harness.fs.local as local_mod

    bad_ctx = Context()
    for bad_cfg in (
        {"cwd": ""},
        {"cwd": 123},
        {"diffBasisMaxBytes": 0},
        {"diffBasisMaxBytes": True},
        {"diffBasisMaxBytes": "1024"},
        {"diffBasisMaxBytes": local_mod._MAX_DIFF_BASIS_BYTES + 1},
    ):
        with pytest.raises(ValueError):
            LocalFileSystem(bad_ctx, bad_cfg)

    ctx = Context()
    ctx.use(LocalFileSystem, {"cwd": str(tmp_path), "diffBasisMaxBytes": 64})
    await ctx.flush()
    fs: LocalFileSystem = ctx.fs

    # process_path_from_host_path rejects relative paths and normalizes absolute paths
    assert fs.process_path_from_host_path("relative/path.txt") is None
    assert fs.process_path_from_host_path(str(tmp_path / "a" / ".." / "b")) == str(
        tmp_path / "b"
    )

    # _version_after_write missing sentinel & _read_bytes_bounded TOCTOU race
    assert local_mod._version_after_write(
        str(tmp_path / "missing-after.txt")
    ).startswith("missing:")
    race_f = tmp_path / "grow.bin"
    race_f.write_bytes(b"123456")
    t_race = await fs.resolve("grow.bin")
    monkeypatch.setattr(local_mod, "_STREAM_CHUNK_BYTES", 2)
    monkeypatch.setattr(
        fs,
        "_require_regular_file",
        lambda _t, _v: ("0:0:1:0:0", 0o644, 1),
    )
    with pytest.raises(FsError) as race_err:
        await fs.read_bytes(t_race, None, 3)
    assert race_err.value.code == "FS_TOO_LARGE"
    monkeypatch.undo()

    # Atomic createIfAbsent link race -> FS_NOT_OBSERVED
    existing_link = tmp_path / "link_race.txt"
    existing_link.write_text("already", encoding="utf-8")
    t_link_race = await fs.resolve("link_race.txt")
    real_probe = local_mod._probe
    monkeypatch.setattr(
        local_mod,
        "_probe",
        lambda p: None if p == t_link_race["targetKey"] else real_probe(p),
    )
    with pytest.raises(FsError) as obs_err:
        await fs.write_text(t_link_race, "new", {"kind": "createIfAbsent"})
    assert obs_err.value.code == "FS_NOT_OBSERVED"
    monkeypatch.undo()

    # Contended lock eviction (refcount increments >0 -> 0 and clears self._locks)
    t_lock = await fs.resolve("contended.txt")
    lock_entered = asyncio.Event()
    allow_release = asyncio.Event()

    async def _hold_first() -> None:
        async with fs._lock_targets(t_lock["targetKey"]):
            lock_entered.set()
            await allow_release.wait()

    async def _wait_second() -> None:
        await lock_entered.wait()
        async with fs._lock_targets(t_lock["targetKey"]):
            assert t_lock["targetKey"] in fs._locks

    first_task = asyncio.create_task(_hold_first())
    second_task = asyncio.create_task(_wait_second())
    await lock_entered.wait()
    await asyncio.sleep(0)
    allow_release.set()
    await asyncio.gather(first_task, second_task)
    assert fs._locks == {}

    # Dispose a context with no active watchers cleanly
    idle_ctx = Context()
    idle_ctx.use(LocalFileSystem, {"cwd": str(tmp_path)})
    await idle_ctx.flush()
    await idle_ctx.fiber.dispose()

    # Polling watch on directory (child add + vanish stat edge), file change, OSError, abort & dispose
    monkeypatch.setattr(local_mod, "_WATCH_POLL_INTERVAL_S", 0.01)
    watch_dir = tmp_path / "watched_dir"
    watch_dir.mkdir()
    dir_t = await fs.resolve("watched_dir")
    dir_events: list[Exception | None] = []
    dir_ctrl = AbortController()
    unwatch_dir = await fs.watch(dir_t, dir_events.append, dir_ctrl.signal)
    await asyncio.sleep(0.025)
    (watch_dir / "item.txt").write_text("hi", encoding="utf-8")
    await asyncio.sleep(0.03)
    assert None in dir_events
    await unwatch_dir()

    # Cover stat(follow_symlinks=False) OSError fallback inside _watch_snapshot on dir
    class _VanishEntry:
        name = "vanished"

        def stat(self, *, follow_symlinks: bool = False) -> Any:
            del follow_symlinks
            raise FileNotFoundError(errno.ENOENT, "gone")

    class _FakeScandir:
        def __enter__(self) -> list[_VanishEntry]:
            return [_VanishEntry()]

        def __exit__(self, *_a: object) -> None:
            return None

    monkeypatch.setattr(local_mod.os, "scandir", lambda _p: _FakeScandir())
    assert local_mod._watch_snapshot(str(watch_dir), True) == (("vanished", None),)
    monkeypatch.undo()

    # Watch file -> detect modification -> stop on non-ENOENT OSError
    monkeypatch.setattr(local_mod, "_WATCH_POLL_INTERVAL_S", 0.01)
    watch_file = tmp_path / "watched.txt"
    watch_file.write_text("v1", encoding="utf-8")
    file_t = await fs.resolve("watched.txt")
    file_events: list[Exception | None] = []
    file_ctrl = AbortController()
    unwatch_file = await fs.watch(file_t, file_events.append, file_ctrl.signal)
    await asyncio.sleep(0.02)
    watch_file.write_text("v2-longer", encoding="utf-8")
    await asyncio.sleep(0.03)
    assert None in file_events

    real_lstat = local_mod.os.lstat

    def _perm_lstat(p: Any) -> Any:
        if str(p) == str(watch_file):
            raise PermissionError(errno.EACCES, "denied")
        return real_lstat(p)

    monkeypatch.setattr(local_mod.os, "lstat", _perm_lstat)
    await asyncio.sleep(0.03)
    assert any(isinstance(ev, PermissionError) for ev in file_events)
    await unwatch_file()
    monkeypatch.undo()

    # Abort signal exits poll cleanly, and context dispose cancels open watchers
    monkeypatch.setattr(local_mod, "_WATCH_POLL_INTERVAL_S", 5.0)
    abort_ctrl = AbortController()
    _unwatch_abort = await fs.watch(file_t, lambda _e: None, abort_ctrl.signal)
    await asyncio.sleep(0)
    abort_ctrl.abort("done")
    await asyncio.sleep(0)

    leak_ctrl = AbortController()
    await fs.watch(file_t, lambda _e: None, leak_ctrl.signal)
    await ctx.fiber.dispose()
    monkeypatch.undo()
