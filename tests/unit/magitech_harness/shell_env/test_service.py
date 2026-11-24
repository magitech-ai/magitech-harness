"""Unit tests for magitech_harness.shell_env."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest
from animas import Context

from magitech_harness.shell_env import (
    MAGITECH_AGENTS_HOME_ENV,
    MAGITECH_HOME_ENV,
    MAGITECH_PROFILE_DIR_KEY,
    MAGITECH_PROFILE_KEY,
    MAGITECH_SESSION_ID_KEY,
    MAGITECH_SHELL_KEY,
    ShellEnvRegistry,
    apply,
    resolve_magitech_agents_home,
    resolve_magitech_home,
)


@dataclass
class _Contributor:
    name: str
    variables: dict[str, dict[str, str]]
    fn: Any

    def resolve(self, execution: Any) -> dict[str, str]:
        return self.fn(execution)


def test_resolve_magitech_home_precedence_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit"
    assert resolve_magitech_home(str(explicit)) == str(explicit.resolve())

    env_dir = tmp_path / "env_home"
    monkeypatch.setenv(MAGITECH_HOME_ENV, str(env_dir))
    assert resolve_magitech_home() == str(env_dir.resolve())

    monkeypatch.delenv(MAGITECH_HOME_ENV, raising=False)
    assert resolve_magitech_home() == str((Path.home() / ".magitech").resolve())

    with pytest.raises(ValueError, match="non-empty string"):
        resolve_magitech_home("   ")


def test_resolve_magitech_agents_home_precedence_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit-agents"
    assert resolve_magitech_agents_home(str(explicit)) == str(explicit.resolve())

    env_dir = tmp_path / "env_agents"
    monkeypatch.setenv(MAGITECH_AGENTS_HOME_ENV, str(env_dir))
    assert resolve_magitech_agents_home() == str(env_dir.resolve())

    monkeypatch.delenv(MAGITECH_AGENTS_HOME_ENV, raising=False)
    assert resolve_magitech_agents_home() == str((Path.home() / ".agents").resolve())

    with pytest.raises(ValueError, match="non-empty string"):
        resolve_magitech_agents_home("   ")


def test_shell_env_registry_registration_conflicts_and_collect(
    tmp_path: Path,
) -> None:
    root = Context()
    with pytest.raises(ValueError, match="must be a string"):
        ShellEnvRegistry(root, {"magitechHome": 123})

    with pytest.raises(ValueError, match="agentsHome must be a string"):
        ShellEnvRegistry(Context(), {"agentsHome": 123})

    root2 = Context()
    reg = apply(root2, {"magitechHome": str(tmp_path / "h")})
    assert root2.shellEnv is reg

    # Validation errors on bad contributors
    with pytest.raises(ValueError, match="name must be non-empty"):
        reg.register(_Contributor("  ", {}, lambda _e: {}))
    with pytest.raises(ValueError, match="declared invalid key"):
        reg.register(
            _Contributor("bad-key", {"OTHER_FOO": {"description": "d"}}, lambda _e: {})
        )
    with pytest.raises(ValueError, match="declared invalid key"):
        reg.register(
            _Contributor(
                "bad-suf", {"MAGITECH_lower": {"description": "d"}}, lambda _e: {}
            )
        )
    with pytest.raises(ValueError, match="cannot own reserved key"):
        reg.register(
            _Contributor(
                "resv", {MAGITECH_SHELL_KEY: {"description": "d"}}, lambda _e: {}
            )
        )
    with pytest.raises(ValueError, match="must describe"):
        reg.register(
            _Contributor(
                "empty-desc", {"MAGITECH_ONE": {"description": " "}}, lambda _e: {}
            )
        )

    dispose_a = reg.register(
        _Contributor(
            "beta",
            {"MAGITECH_Z": {"description": "z var"}},
            lambda _e: {"MAGITECH_Z": "vz"},
        )
    )
    dispose_b = reg.register(
        _Contributor(
            "alpha",
            {"MAGITECH_A": {"description": "a var"}},
            lambda _e: {"MAGITECH_A": "va"},
        )
    )
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_Contributor("alpha", {}, lambda _e: {}))
    with pytest.raises(ValueError, match="already owned"):
        reg.register(
            _Contributor("gamma", {"MAGITECH_A": {"description": "dup"}}, lambda _e: {})
        )

    assert [item["key"] for item in reg.list()] == ["MAGITECH_A", "MAGITECH_Z"]

    # Collect with no execution / no agent / no profileContext
    base_snap = reg.collect(None)
    assert isinstance(base_snap, type(reg.collect()))
    assert dict(base_snap) == {
        "MAGITECH_A": "va",
        "MAGITECH_AGENTS_HOME": str((Path.home() / ".agents").resolve()),
        "MAGITECH_HOME": str((tmp_path / "h").resolve()),
        "MAGITECH_SHELL": "1",
        "MAGITECH_Z": "vz",
    }
    with pytest.raises(TypeError):
        base_snap["MAGITECH_NEW"] = "x"  # type: ignore[index]

    # Collect with session header dict, session_id attr, agent.id attr, and profileContext (mapping & obj)
    class _SessHeader:
        header: ClassVar[dict[str, str]] = {"id": "sess-hdr"}

    class _SessAttr:
        session_id = "sess-sid"

    class _Agent:
        def __init__(self, session: Any, aid: str | None = None) -> None:
            self.session = session
            self.id = aid

    root2.provide("profileContext", {"name": "dev", "dir": "/tmp/dev"})
    snap_hdr = reg.collect({"agent": _Agent(_SessHeader())})
    assert snap_hdr[MAGITECH_SESSION_ID_KEY] == "sess-hdr"
    assert snap_hdr[MAGITECH_PROFILE_KEY] == "dev"
    assert snap_hdr[MAGITECH_PROFILE_DIR_KEY] == "/tmp/dev"

    class _ProfObj:
        name = "prod"
        dir = "/tmp/prod"

    root2._services["profileContext"] = _ProfObj()
    assert reg.collect(_Agent(_SessAttr()))  # no agent attr -> treated as no agent

    class _ExecObj:
        agent = _Agent(_SessAttr())

    assert reg.collect(_ExecObj())[MAGITECH_SESSION_ID_KEY] == "sess-sid"
    assert (
        reg.collect({"agent": _Agent(None, "aid-1")})[MAGITECH_SESSION_ID_KEY]
        == "aid-1"
    )
    assert (
        reg.collect({"agent": _Agent(type("_BareSess", (), {})(), "aid-2")})[
            MAGITECH_SESSION_ID_KEY
        ]
        == "aid-2"
    )

    # Bad profileContext values ignored cleanly
    root2._services["profileContext"] = object()
    assert MAGITECH_PROFILE_KEY not in reg.collect({})

    # Runtime validation on undeclared / non-string returns
    reg.register(
        _Contributor(
            "bad-ret",
            {"MAGITECH_X": {"description": "d"}},
            lambda _e: {"MAGITECH_Y": "1"},
        )
    )
    with pytest.raises(ValueError, match="undeclared key"):
        reg.collect({})
    # Dispose alpha/beta cleanly; register bad value contributor in child scope
    dispose_b()
    dispose_a()
    assert reg.list() == [
        {"contributor": "bad-ret", "key": "MAGITECH_X", "description": "d"}
    ]

    reg2 = apply(Context(), {"magitechHome": str(tmp_path / "h2")})
    reg2.register(
        _Contributor(
            "bad-type",
            {"MAGITECH_X": {"description": "d"}},
            lambda _e: {"MAGITECH_X": 42},
        )
    )
    with pytest.raises(TypeError, match="non-string value"):
        reg2.collect({})
