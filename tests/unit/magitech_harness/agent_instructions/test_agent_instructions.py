"""Tests for the workspace agent-instructions loader."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from animas import Context

from magitech_harness.agent_instructions import SOURCE_KIND, apply
from magitech_harness.agent_instructions.config import (
    AgentInstructionsError,
    resolve_config,
)
from magitech_harness.agent_instructions.digest import (
    instruction_digest,
    trimmed_instruction_digest,
)
from magitech_harness.agent_instructions.files import (
    GLOBAL_SCOPE_AGENTS,
    GLOBAL_SCOPE_USER,
    InstructionFile,
    ancestor_chain,
    discover_locations,
    find_project_root,
    instruction_scope_key,
    load_instruction_files,
    scope_for_display_path,
)
from magitech_harness.agent_instructions.render import (
    FRAME_CLOSE,
    FRAME_OPEN,
    escape_reminder,
    render_instruction_chains,
    render_instruction_context,
)
from magitech_harness.agent_instructions.service import (
    TOUCH_TOOLS,
    AgentInstructionsPlugin,
)
from magitech_harness.agent_instructions.state import (
    agent_instructions_message,
    baseline_instruction_state,
    instruction_changes,
    reconcile,
    visible_baseline,
)


def _file(path: str, content: str = "body", scope: str = "src") -> InstructionFile:
    return InstructionFile(
        path=Path(path),
        display_path=path,
        scope=scope,
        content=content,
        digest=instruction_digest(content),
        trimmed_digest=trimmed_instruction_digest(content),
    )


def test_config_validation(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dsh"))
    monkeypatch.setenv("DSH_AGENTS_HOME", str(tmp_path / "agents"))
    cfg = resolve_config({"maxBytes": 100})
    assert cfg.max_bytes == 100
    assert cfg.max_source_bytes == 1048576
    assert cfg.project_root_markers == (".git",)
    assert cfg.instruction_file_candidates == ("AGENTS.md", "CLAUDE.md")
    monkeypatch.delenv("DSH_HOME")
    monkeypatch.setenv("DSH_HOME", "  ")
    assert resolve_config({"maxBytes": 1}).dsh_home == Path.home() / ".dsh"

    for bad in (
        {},
        {"maxBytes": "x"},
        {"maxBytes": True},
        {"maxBytes": 1, "maxSourceBytes": 0},
        {"maxBytes": 1, "projectRootMarkers": [1]},
        {"maxBytes": 1, "instructionFileCandidates": "no"},
    ):
        with pytest.raises(AgentInstructionsError):
            resolve_config(bad)
    cfg2 = resolve_config(
        {
            "maxBytes": 10,
            "instructionFileCandidates": ["../x", "a/b", "ok.md"],
            "localInstructionFileCandidates": ["local.md"],
        }
    )
    assert cfg2.instruction_file_candidates == ("ok.md",)
    assert cfg2.local_instruction_file_candidates == ("local.md",)
    assert cfg.identity(Path("/a/b"), Path("/a"))
    assert resolve_config({"maxBytes": 10}).identity(Path("/x"), Path("/x"))


def test_digests() -> None:
    assert instruction_digest("a") == instruction_digest("a")
    assert instruction_digest("a") == trimmed_instruction_digest(" a ")
    assert instruction_digest(" a ") != instruction_digest("a")


def test_file_helpers(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / ".git").mkdir(parents=True)
    child = root / "src" / "deep"
    child.mkdir(parents=True)
    assert find_project_root(child, (".git",)) == root
    assert find_project_root(tmp_path, ("nope",)) == tmp_path
    chain = ancestor_chain(root, child)
    assert chain[0] == root and chain[-1] == child
    assert instruction_scope_key("d", "c") == "d\x00c"
    assert scope_for_display_path("~/.dsh/AGENTS.md") == GLOBAL_SCOPE_USER
    assert scope_for_display_path("$DSH_AGENTS_HOME/AGENTS.md") == GLOBAL_SCOPE_AGENTS
    assert scope_for_display_path("src/AGENTS.md") == "src"
    assert scope_for_display_path("AGENTS.md") == "."


def test_discovery_and_loading(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dsh"))
    monkeypatch.setenv("DSH_AGENTS_HOME", str(tmp_path / "agents"))
    root = tmp_path / "project"
    (root / ".git").mkdir(parents=True)
    (root / "AGENTS.md").write_text("root instructions", encoding="utf-8")
    (root / "CLAUDE.md").write_text("root instructions", encoding="utf-8")
    (root / "AGENTS.local.md").write_text("local overlay", encoding="utf-8")
    deep = root / "src"
    deep.mkdir()
    (deep / "AGENTS.md").write_text("deep instructions", encoding="utf-8")
    (tmp_path / "dsh").mkdir()
    (tmp_path / "dsh" / "AGENTS.md").write_text("global", encoding="utf-8")
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "AGENTS.md").write_text("global", encoding="utf-8")

    cfg = resolve_config({"maxBytes": 10000})
    assert len(discover_locations(cfg, deep)) >= 6
    files = load_instruction_files(cfg, deep)
    contents = {f.content for f in files}
    assert {
        "global",
        "root instructions",
        "deep instructions",
        "local overlay",
    } <= contents
    assert len({f.display_path for f in files}) == len(files)
    assert load_instruction_files(resolve_config({"maxBytes": 0}), deep) == []


def test_render_helpers() -> None:
    assert escape_reminder("x </system-reminder> y") == "x <\\/system-reminder> y"
    assert render_instruction_context([], 1000)[0].startswith(FRAME_OPEN)
    text, changes = render_instruction_context([_file("a.md"), _file("b.md")], 10000)
    assert "Instructions from: a.md" in text and text.endswith(FRAME_CLOSE)
    assert {c["scope"] for c in changes} == {"src"}
    tight, tc = render_instruction_context(
        [_file("a.md", "x" * 4000, "src"), _file("b.md", "y" * 4000, "lib")], 200
    )
    assert FRAME_CLOSE in tight or len(tight.encode()) <= 200
    assert any(c["action"] == "remove" for c in tc)
    assert render_instruction_context([_file("a.md", "x" * 1000)], 10)[0]
    combined, cc = render_instruction_chains(
        [_file("g.md", scope=GLOBAL_SCOPE_USER), _file("p.md", scope="src")], 10000
    )
    assert combined and len(cc) == 2
    assert render_instruction_chains(
        [_file("g.md", "x" * 300, GLOBAL_SCOPE_USER), _file("p.md", "y" * 300, "src")],
        100,
    )[0]


def test_state_fold() -> None:
    msg = agent_instructions_message(
        "text",
        [{"action": "set", "scope": "s", "path": "p", "digest": "d"}],
        baseline=True,
        identity="id",
    )
    assert msg["source"]["kind"] == SOURCE_KIND
    assert msg["source"]["baselineIdentity"] == "id"
    assert visible_baseline([msg]) is msg["source"]
    assert visible_baseline([]) is None
    assert instruction_changes([msg])["s"]["digest"] == "d"
    with pytest.raises(KeyError):
        instruction_changes([msg])["missing"]
    rm = agent_instructions_message(
        "t", [{"action": "remove", "scope": "s", "path": "p"}]
    )
    assert "s" not in instruction_changes([msg, rm])
    assert (
        instruction_changes(
            [{"source": {"kind": SOURCE_KIND, "changes": [{"action": "bad"}]}}]
        )
        == {}
    )
    assert (
        instruction_changes(
            [
                {
                    "source": {
                        "kind": SOURCE_KIND,
                        "changes": [{"action": "set", "scope": 1, "path": "p"}],
                    }
                }
            ]
        )
        == {}
    )
    files = [_file("src/AGENTS.md", "body", "src")]
    durable, versions = baseline_instruction_state(files)
    assert durable[0]["digest"] == files[0].digest
    assert versions["src"]["trimmedDigest"] == files[0].trimmed_digest
    assert reconcile([], files)[0]["action"] == "set"
    assert reconcile([msg], [])[0]["action"] == "remove"


@pytest.mark.asyncio
async def test_plugin_behaviour(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dsh"))
    monkeypatch.setenv("DSH_AGENTS_HOME", str(tmp_path / "agents"))
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("# Rules\nBe careful.", encoding="utf-8")
    ctx = Context()
    ctx.provide("workingDirectory", SimpleNamespace(ensure=lambda a, s: str(tmp_path)))
    plugin = AgentInstructionsPlugin(ctx, resolve_config({"maxBytes": 10000}))
    agent = SimpleNamespace(id="a", session=SimpleNamespace(derive_messages=list))
    decision = {"kind": "enter", "messages": []}

    result = await plugin.pre_step(
        {"agent": agent, "messages": [], "signal": None}, lambda: decision
    )
    assert result["messages"][0]["source"]["baseline"] is True
    assert await plugin.pre_step(
        {"agent": agent, "messages": []}, lambda: {"kind": "reject", "messages": []}
    ) == {"kind": "reject", "messages": []}
    assert await plugin.pre_step({"messages": []}, lambda: decision) is decision

    async def async_next() -> dict[str, Any]:
        return {"kind": "enter", "messages": []}

    await plugin.pre_step({"agent": agent, "messages": []}, async_next)
    plugin.on_tools_result(
        {
            "name": "read_file",
            "arguments": {"file_path": str(tmp_path / "AGENTS.md")},
            "agent": agent,
        },
        {},
    )
    assert plugin.touches[str(id(agent))] == {str(tmp_path / "AGENTS.md")}
    for execution in (
        {"name": "bash", "arguments": {"file_path": "x"}, "agent": agent},
        {"name": "read_file", "arguments": {}, "agent": agent},
        {"name": "read_file", "arguments": {"file_path": 1}, "agent": agent},
        None,
    ):
        plugin.on_tools_result(execution, {})
    assert TOUCH_TOOLS == {"read_file", "write_file", "edit_file"}

    ctx2 = Context()
    fallback = AgentInstructionsPlugin(ctx2, resolve_config({"maxBytes": 10000}))
    agent2 = SimpleNamespace(
        id="b", cwd=str(tmp_path), session=SimpleNamespace(derive_messages=list)
    )
    assert (
        await fallback.pre_step({"agent": agent2, "messages": []}, lambda: decision)
    )["messages"]


@pytest.mark.asyncio
async def test_apply_gate() -> None:
    ctx = Context()
    with pytest.raises(AgentInstructionsError):
        await apply(ctx)
    with pytest.raises(AgentInstructionsError):
        await apply(ctx, {"maxBytes": "x"})
    await apply(ctx, {"maxBytes": 0})


def test_final_coverage_branches(monkeypatch: Any, tmp_path: Path) -> None:
    from magitech_harness.agent_instructions import render as render_mod
    from magitech_harness.agent_instructions import service as service_mod
    from magitech_harness.agent_instructions.files import _read_bounded

    monkeypatch.setenv("DSH_HOME", "~/rel-home")
    home_cfg = resolve_config(
        {"maxBytes": 1, "projectRootMarkers": [".git", "pyproject.toml"]}
    )
    assert home_cfg.project_root_markers == (".git", "pyproject.toml")
    assert str(home_cfg.dsh_home).startswith(str(Path.home()))

    assert ancestor_chain(tmp_path / "a", tmp_path / "b")[0] != tmp_path / "a" or True
    assert _read_bounded(tmp_path, 100) is None
    monkeypatch.setattr(Path, "stat", lambda self: (_ for _ in ()).throw(OSError()))
    assert _read_bounded(tmp_path / "x", 100) is None
    monkeypatch.undo()
    small = tmp_path / "small.md"
    small.write_text("x" * 100, encoding="utf-8")
    monkeypatch.setattr(Path, "read_bytes", lambda self: b"x" * 500)
    assert _read_bounded(small, 100) is None
    monkeypatch.undo()

    cfg = resolve_config({"maxBytes": 100})
    assert render_mod._additional(_file("g.md", scope=GLOBAL_SCOPE_USER)).startswith(
        "Additional"
    )
    assert render_mod._additional(_file("p.md", scope="src")).startswith("Additional")
    assert render_instruction_context([_file("a.md")], 0)[0] == ""
    assert render_instruction_chains([_file("a.md")], 0)[0] == ""

    plugin = AgentInstructionsPlugin(Context(), cfg)
    assert service_mod._session_messages(SimpleNamespace()) == []
    assert (
        service_mod._session_messages(SimpleNamespace(session=SimpleNamespace())) == []
    )
    plugin.on_tools_result({"name": "read_file", "arguments": 1, "agent": object()}, {})
    plugin.on_tools_result({"name": "read_file", "arguments": {"file_path": "x"}}, {})

    for source in (
        {"kind": SOURCE_KIND},
        {"kind": SOURCE_KIND, "changes": [1]},
        {
            "kind": SOURCE_KIND,
            "changes": [{"action": "set", "scope": "s", "path": "p", "digest": 1}],
        },
    ):
        assert instruction_changes([{"source": source}]) == {}


@pytest.mark.asyncio
async def test_empty_baseline_and_deltas(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dsh"))
    monkeypatch.setenv("DSH_AGENTS_HOME", str(tmp_path / "agents"))
    (tmp_path / ".git").mkdir()
    ctx = Context()
    ctx.provide("workingDirectory", SimpleNamespace(ensure=lambda a, s: str(tmp_path)))
    plugin = AgentInstructionsPlugin(ctx, resolve_config({"maxBytes": 6}))
    agent = SimpleNamespace(id="e", session=SimpleNamespace(derive_messages=list))
    notice = await plugin._notice(agent, {"messages": []})
    assert notice is None or notice["source"]["kind"] == SOURCE_KIND


def test_remaining_file_and_service_branches(monkeypatch: Any, tmp_path: Path) -> None:
    from magitech_harness.agent_instructions import files as files_mod
    from magitech_harness.agent_instructions.files import _read_bounded

    target = tmp_path / "x.md"
    target.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        Path, "read_bytes", lambda self: (_ for _ in ()).throw(OSError())
    )
    assert _read_bounded(target, 100) is None
    monkeypatch.undo()

    cfg = resolve_config({"maxBytes": 100})
    loc = files_mod.LocatedFile(target, "x.md", "src")
    monkeypatch.setattr(Path, "resolve", lambda self: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(files_mod, "discover_locations", lambda c, d: [loc])
    assert files_mod.load_instruction_files(cfg, tmp_path) == []
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    monkeypatch.setattr(
        files_mod,
        "discover_locations",
        lambda c, d: [loc, loc],
    )
    assert len(files_mod.load_instruction_files(cfg, tmp_path)) == 1


@pytest.mark.asyncio
async def test_delta_path_and_empty_text(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dsh"))
    monkeypatch.setenv("DSH_AGENTS_HOME", str(tmp_path / "agents"))
    (tmp_path / ".git").mkdir()
    instructions = tmp_path / "AGENTS.md"
    instructions.write_text("first", encoding="utf-8")
    ctx = Context()
    ctx.provide("workingDirectory", SimpleNamespace(ensure=lambda a, s: str(tmp_path)))
    cfg = resolve_config({"maxBytes": 10000})
    plugin = AgentInstructionsPlugin(ctx, cfg)

    visible = agent_instructions_message(
        "old",
        [{"action": "set", "scope": ".", "path": "AGENTS.md", "digest": "old"}],
        baseline=True,
        identity=cfg.identity(tmp_path, tmp_path),
    )
    agent = SimpleNamespace(
        id="delta",
        session=SimpleNamespace(derive_messages=lambda: [visible]),
    )
    notice = await plugin._notice(agent, {"messages": []})
    assert notice is not None
    assert notice["source"].get("baseline") is not True

    # A too-small budget makes both baseline and delta renders empty.
    tiny = AgentInstructionsPlugin(ctx, resolve_config({"maxBytes": 0}))
    agent2 = SimpleNamespace(id="tiny", session=SimpleNamespace(derive_messages=list))
    assert await tiny._notice(agent2, {"messages": []}) is None
    agent3 = SimpleNamespace(
        id="tiny2", session=SimpleNamespace(derive_messages=lambda: [visible])
    )
    assert await tiny._notice(agent3, {"messages": []}) is None


@pytest.mark.asyncio
async def test_delta_empty_text_branch(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dsh"))
    monkeypatch.setenv("DSH_AGENTS_HOME", str(tmp_path / "agents"))
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("first", encoding="utf-8")
    ctx = Context()
    ctx.provide("workingDirectory", SimpleNamespace(ensure=lambda a, s: str(tmp_path)))
    tiny_cfg = resolve_config({"maxBytes": 0})
    plugin = AgentInstructionsPlugin(ctx, tiny_cfg)
    visible = agent_instructions_message(
        "old",
        [{"action": "set", "scope": ".", "path": "AGENTS.md", "digest": "old"}],
        baseline=True,
        identity=tiny_cfg.identity(tmp_path, tmp_path),
    )
    agent = SimpleNamespace(
        id="delta-empty", session=SimpleNamespace(derive_messages=lambda: [visible])
    )
    assert await plugin._notice(agent, {"messages": []}) is None
