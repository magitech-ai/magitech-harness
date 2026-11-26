"""Tests for the permission-preset domain: catalog, fold, and write paths."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from animas import Context

from magitech_harness import create_harness
from magitech_harness.permission import (
    AUTO_PRESET,
    AUTO_PRESET_SPEC,
    CUSTOM_PRESET,
    DEFAULT_PRESETS,
    RESERVED_PRESETS,
    PermissionPresetError,
    PermissionPresetService,
    PresetSpec,
)
from magitech_harness.permission.service import (
    _apply_event,
    _init_state,
    _parse_presets,
    _require_name,
    _validate_selection,
    _validate_state,
    apply,
)
from magitech_harness.session.events import EventType, SessionEvent


def _agent(session: Any) -> SimpleNamespace:
    return SimpleNamespace(session=session)


def _state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "preset": None,
        "sandbox": None,
        "approval": None,
        "seeded": False,
    }
    state.update(overrides)
    return state


def _event(event_type: str, payload: dict[str, Any]) -> SessionEvent:
    return SessionEvent(
        id="s:0",
        session_id="s",
        type=event_type,
        timestamp=0.0,
        payload=payload,
        seq=0,
    )


def _types(session: Any) -> list[str]:
    return [event.type for event in session.events]


@pytest.fixture
def harness() -> Any:
    return create_harness()


# -- pure: presets and names -----------------------------------------------


def test_preset_specs_and_constants() -> None:
    spec = PresetSpec("read-only", "ask", "Safe", "description")
    assert (spec.sandbox, spec.approval, spec.name, spec.description) == (
        "read-only",
        "ask",
        "Safe",
        "description",
    )
    assert RESERVED_PRESETS == {"custom", "auto"}
    assert AUTO_PRESET_SPEC == ("danger-full-access", "ask")
    assert list(DEFAULT_PRESETS) == ["workspace-write", "danger-full-access"]
    parsed = _parse_presets(DEFAULT_PRESETS)
    assert parsed["workspace-write"].name == "workspace-write"
    assert parsed["danger-full-access"].approval == "never"


def test_require_name() -> None:
    assert _require_name("value", "subject") == "value"
    for value in (None, 1, ""):
        with pytest.raises(PermissionPresetError, match="non-empty string"):
            _require_name(value, "subject")


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"custom": {"sandbox": "read-only", "approval": "ask"}},
        {"auto": {"sandbox": "read-only", "approval": "ask"}},
        {1: {"sandbox": "read-only", "approval": "ask"}},
        {"": {"sandbox": "read-only", "approval": "ask"}},
        {"p": "not-a-mapping"},
        {"p": {"sandbox": "read-only", "approval": "ask", "extra": True}},
        {"p": {"sandbox": "bogus", "approval": "ask"}},
        {"p": {"sandbox": "read-only", "approval": "bogus"}},
        {"p": {"sandbox": "read-only", "approval": "ask", "name": 1}},
        {"p": {"sandbox": "read-only", "approval": "ask", "name": ""}},
        {"p": {"sandbox": "read-only", "approval": "ask", "description": 1}},
    ],
)
def test_parse_presets_rejects_invalid_tables(value: Any) -> None:
    with pytest.raises(PermissionPresetError):
        _parse_presets(value)


def test_parse_presets_accepts_presentation() -> None:
    parsed = _parse_presets(
        {
            "safe": {
                "sandbox": "read-only",
                "approval": "ask",
                "name": "Safe",
                "description": "Read only.",
            }
        }
    )
    assert parsed["safe"] == PresetSpec("read-only", "ask", "Safe", "Read only.")


# -- pure: projection fold --------------------------------------------------


def test_validate_state() -> None:
    assert _validate_state(_state()) == _state()
    with pytest.raises(TypeError, match="object"):
        _validate_state([])
    with pytest.raises(ValueError, match="fields"):
        _validate_state({"preset": None})
    with pytest.raises(ValueError, match="fields"):
        _validate_state({**_state(), "extra": 1})
    with pytest.raises(ValueError, match="preset"):
        _validate_state(_state(preset=1))
    with pytest.raises(ValueError, match="preset"):
        _validate_state(_state(preset=""))
    with pytest.raises(ValueError, match="sandbox"):
        _validate_state(_state(sandbox="bogus"))
    with pytest.raises(ValueError, match="approval"):
        _validate_state(_state(approval="bogus"))
    with pytest.raises(TypeError, match="seeded"):
        _validate_state(_state(seeded=1))


def test_init_state() -> None:
    assert _init_state({"id": "s"}, 0) == _state()


def test_apply_event_knobs() -> None:
    base = _state()
    preset = _apply_event(base, _event(EventType.PERMISSION_PRESET, {"preset": "p"}))
    assert preset["preset"] == "p"
    assert preset is not base
    assert (
        _apply_event(preset, _event(EventType.PERMISSION_PRESET, {"preset": "p"}))
        is preset
    )
    sandbox = _apply_event(base, _event(EventType.SANDBOX_MODE, {"mode": "read-only"}))
    assert sandbox["sandbox"] == "read-only"
    assert (
        _apply_event(sandbox, _event(EventType.SANDBOX_MODE, {"mode": "read-only"}))
        is sandbox
    )
    approval = _apply_event(
        base, _event(EventType.APPROVAL_POLICY, {"policy": "never"})
    )
    assert approval["approval"] == "never"
    assert (
        _apply_event(approval, _event(EventType.APPROVAL_POLICY, {"policy": "never"}))
        is approval
    )
    seeded = _apply_event(base, _event(EventType.SESSION_END_SEED, {}))
    assert seeded["seeded"] is True
    assert _apply_event(seeded, _event(EventType.SESSION_END_SEED, {})) is seeded
    other = _event(EventType.TURN_START, {})
    assert _apply_event(base, other) is base


@pytest.mark.parametrize(
    ("event_type", "payload"),
    [
        (EventType.PERMISSION_PRESET, {}),
        (EventType.PERMISSION_PRESET, {"preset": ""}),
        (EventType.PERMISSION_PRESET, {"preset": 1}),
        (EventType.SANDBOX_MODE, {"mode": "bogus"}),
        (EventType.SANDBOX_MODE, {}),
        (EventType.APPROVAL_POLICY, {"policy": "bogus"}),
        (EventType.APPROVAL_POLICY, {}),
    ],
)
def test_apply_event_rejects_invalid_payloads(
    event_type: str, payload: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        _apply_event(_state(), _event(event_type, payload))


def test_validate_selection() -> None:
    assert _validate_selection({"currentValue": "read-only"}) == {
        "currentValue": "read-only"
    }
    with pytest.raises(ValueError, match="object"):
        _validate_selection([])
    with pytest.raises(ValueError, match="currentValue"):
        _validate_selection({})
    with pytest.raises(ValueError, match="currentValue"):
        _validate_selection({"currentValue": ""})
    with pytest.raises(ValueError, match="currentValue"):
        _validate_selection({"currentValue": 1})


# -- service: configuration ------------------------------------------------


def test_default_catalog(harness: Any) -> None:
    service: PermissionPresetService = harness.permissionPresets
    assert service.names == ["read-only", "workspace-write", "danger-full-access"]
    assert service.default_preset == "workspace-write"
    assert service.catalog() == {
        "options": [
            {"value": "read-only", "name": "read-only"},
            {"value": "workspace-write", "name": "workspace-write"},
            {"value": "danger-full-access", "name": "danger-full-access"},
        ],
        "defaultOptions": [
            {"value": "read-only", "name": "read-only"},
            {"value": "workspace-write", "name": "workspace-write"},
            {"value": "danger-full-access", "name": "danger-full-access"},
        ],
        "defaultPreset": "workspace-write",
    }
    assert service.option_of(CUSTOM_PRESET) == {
        "value": "custom",
        "name": "Custom",
        "description": "Current sandbox and approval settings do not match a preset.",
    }
    with pytest.raises(PermissionPresetError, match="unknown preset"):
        service.option_of("missing")


def test_custom_table_and_default(harness: Any) -> None:
    service = PermissionPresetService(
        harness,
        {
            "presets": {
                "safe": {
                    "sandbox": "read-only",
                    "approval": "ask",
                    "name": "Safe",
                    "description": "Read only.",
                },
                "yolo": {"sandbox": "danger-full-access", "approval": "never"},
            },
            "defaultPreset": "safe",
        },
    )
    assert service.names == ["safe", "yolo"]
    assert service.default_preset == "safe"
    assert service.option_of("safe") == {
        "value": "safe",
        "name": "Safe",
        "description": "Read only.",
    }
    assert service.option_of("yolo") == {"value": "yolo", "name": "yolo"}
    assert service.catalog()["defaultOptions"] == [
        {"value": "safe", "name": "Safe", "description": "Read only."},
        {"value": "yolo", "name": "yolo"},
    ]


def test_schema_default_table(harness: Any) -> None:
    service = PermissionPresetService(harness)
    assert service.names == ["workspace-write", "danger-full-access"]
    assert service.option_of("workspace-write")["description"] is not None


def test_apply_mounts_service() -> None:
    ctx = Context()
    ctx.provide("shell", SimpleNamespace(sandbox_mode="workspace-write"))
    ctx.provide("approval", SimpleNamespace(policy="ask"))
    assert isinstance(apply(ctx), PermissionPresetService)
    assert "shell" in apply.inject


@pytest.mark.parametrize(
    "config",
    [
        {"extra": 1},
        {"presets": {}},
        {"defaultPreset": ""},
        {"defaultPreset": 1},
        {"defaultPreset": CUSTOM_PRESET},
        {"defaultPreset": "missing"},
    ],
)
def test_config_rejections(harness: Any, config: dict[str, Any]) -> None:
    with pytest.raises(PermissionPresetError):
        PermissionPresetService(harness, config)


def test_unconfined_shell_is_refused(harness: Any) -> None:
    shell = SimpleNamespace(sandbox_mode=None)
    ctx = Context()
    ctx.provide("shell", shell)
    ctx.provide("approval", harness.approval)
    with pytest.raises(PermissionPresetError, match="does not confine"):
        PermissionPresetService(ctx)


def test_composed_defaults_reject_invalid_state() -> None:
    shell = SimpleNamespace(sandbox_mode="workspace-write")
    approval = SimpleNamespace(policy="ask")
    ctx = Context()
    ctx.provide("shell", shell)
    ctx.provide("approval", approval)
    service = PermissionPresetService(ctx)
    shell.sandbox_mode = None
    with pytest.raises(PermissionPresetError, match="does not confine"):
        service._composed_sandbox()
    shell.sandbox_mode = "workspace-write"
    approval.policy = "bogus"
    with pytest.raises(PermissionPresetError, match="default policy"):
        service._composed_approval()


def test_state_before_init_is_refused(harness: Any) -> None:
    service = PermissionPresetService(harness)
    with pytest.raises(PermissionPresetError, match="not ready"):
        service.current(harness.sessions.create("not-ready"))


# -- service: derive -------------------------------------------------------


def test_derive_prefers_recorded_selection_and_custom(harness: Any) -> None:
    service = PermissionPresetService(
        harness,
        {
            "presets": {
                "a": {"sandbox": "workspace-write", "approval": "ask"},
                "b": {"sandbox": "workspace-write", "approval": "ask"},
                "ro": {"sandbox": "read-only", "approval": "ask"},
            },
            "defaultPreset": "a",
        },
    )
    assert service._derive(_state(preset="b")) == "b"
    assert service._derive(_state(preset="missing")) == "a"
    assert (
        service._derive(_state(sandbox="read-only", approval="never")) == CUSTOM_PRESET
    )
    assert service._derive(_state(sandbox="read-only")) == "ro"


def test_derive_auto(harness: Any) -> None:
    service = PermissionPresetService(
        harness,
        {
            "presets": {
                "workspace-write": {"sandbox": "workspace-write", "approval": "ask"}
            },
            "defaultPreset": "workspace-write",
        },
    )
    with pytest.raises(PermissionPresetError):
        service.resolve(AUTO_PRESET)
    dispose = service.register_auto(lambda: None)
    assert service.resolve(AUTO_PRESET) == PresetSpec("danger-full-access", "ask")
    assert (
        service._derive(
            _state(
                preset=AUTO_PRESET,
                sandbox="danger-full-access",
                approval="ask",
            )
        )
        == AUTO_PRESET
    )
    assert (
        service._derive(
            _state(
                preset=AUTO_PRESET,
                sandbox="danger-full-access",
                approval="never",
            )
        )
        == AUTO_PRESET
    )
    assert (
        service._derive(_state(preset=AUTO_PRESET, sandbox="read-only"))
        == CUSTOM_PRESET
    )
    dispose()
    assert AUTO_PRESET not in service.names


def test_register_auto_catalog_and_conflict(harness: Any) -> None:
    service: PermissionPresetService = harness.permissionPresets
    seen: list[str] = []
    dispose = service.register_auto(lambda: seen.append("admit"))
    assert service.names[-1] == AUTO_PRESET
    assert service.catalog()["options"][-1] == {"value": "auto", "name": "auto"}
    assert service.catalog()["defaultOptions"][-1]["value"] == "danger-full-access"
    with pytest.raises(PermissionPresetError, match="already registered"):
        service.register_auto(lambda: None)
    dispose()
    assert AUTO_PRESET not in service.names
    dispose()


def test_catalog_changed_listener_failure_is_contained(harness: Any) -> None:
    service: PermissionPresetService = harness.permissionPresets

    def explode() -> None:
        raise RuntimeError("listener failure")

    harness.on("permission-presets/catalog-changed", explode)
    dispose = service.register_auto(lambda: None)
    dispose()


# -- service: writes -------------------------------------------------------


def test_set_writes_only_changed_knobs(harness: Any) -> None:
    service: PermissionPresetService = harness.permissionPresets
    session = harness.sessions.create("writes")
    service.set(session, "workspace-write")
    assert _types(session) == []
    service.set(session, "read-only")
    assert _types(session) == ["permission/preset", "sandbox/mode"]
    assert session.events[0].payload == {"preset": "read-only"}
    assert session.events[1].payload == {"mode": "read-only"}
    service.set(session, "read-only")
    assert _types(session) == ["permission/preset", "sandbox/mode"]
    service.set(session, "danger-full-access")
    assert _types(session) == [
        "permission/preset",
        "sandbox/mode",
        "permission/preset",
        "sandbox/mode",
        "approval/policy",
    ]
    assert session.events[-1].payload == {"policy": "never"}
    with pytest.raises(PermissionPresetError, match="unknown preset"):
        service.set(session, "missing")


def test_set_preset_by_id(harness: Any) -> None:
    service: PermissionPresetService = harness.permissionPresets
    session = harness.sessions.create("by-id")
    assert service.set_preset("by-id", "danger-full-access") == {
        "currentValue": "danger-full-access"
    }
    assert service.current(session) == "danger-full-access"
    with pytest.raises(PermissionPresetError, match="no live session"):
        service.set_preset("missing", "read-only")
    with pytest.raises(PermissionPresetError, match="unknown preset"):
        service.set_preset("by-id", "missing")


def test_apply_auto_calls_admission(harness: Any) -> None:
    service: PermissionPresetService = harness.permissionPresets
    admitted: list[str] = []
    dispose = service.register_auto(lambda: admitted.append("yes"))
    session = harness.sessions.create("auto")
    service.set(session, AUTO_PRESET)
    assert admitted == ["yes"]
    assert service.current(session) == AUTO_PRESET
    assert _types(session) == ["permission/preset", "sandbox/mode"]
    assert session.events[1].payload == {"mode": "danger-full-access"}
    service.set(session, AUTO_PRESET)
    assert admitted == ["yes", "yes"]
    assert _types(session) == ["permission/preset", "sandbox/mode"]
    dispose()


# -- service: initial pinning ----------------------------------------------


def test_pin_fresh_session(harness: Any) -> None:
    service: PermissionPresetService = harness.permissionPresets
    session = harness.sessions.create("fresh")
    service.pin_initial_permission(session)
    assert _types(session) == [
        "permission/preset",
        "sandbox/mode",
        "approval/policy",
    ]
    assert [event.payload for event in session.events] == [
        {"preset": "workspace-write"},
        {"mode": "workspace-write"},
        {"policy": "ask"},
    ]
    service.pin_initial_permission(session)
    assert len(session.events) == 3


def test_pin_partial_session(harness: Any) -> None:
    service: PermissionPresetService = harness.permissionPresets
    session = harness.sessions.create("partial")
    session.append(EventType.SANDBOX_MODE, {"mode": "read-only"})
    service.pin_initial_permission(session)
    assert _types(session) == [
        "sandbox/mode",
        "permission/preset",
        "approval/policy",
    ]
    assert session.events[1].payload == {"preset": "read-only"}


def test_pin_seeded_session_keeps_effective_values(harness: Any) -> None:
    service: PermissionPresetService = harness.permissionPresets
    session = harness.sessions.create("seeded")
    session.append(EventType.SESSION_END_SEED, {})
    service.pin_initial_permission(session)
    assert _types(session) == [
        "session/end-seed",
        "permission/preset",
        "sandbox/mode",
        "approval/policy",
    ]


def test_pin_seeded_custom_session(harness: Any) -> None:
    service: PermissionPresetService = harness.permissionPresets
    session = harness.sessions.create("custom")
    session.append(EventType.SESSION_END_SEED, {})
    session.append(EventType.SANDBOX_MODE, {"mode": "read-only"})
    session.append(EventType.APPROVAL_POLICY, {"policy": "never"})
    service.pin_initial_permission(session)
    assert _types(session) == [
        "session/end-seed",
        "sandbox/mode",
        "approval/policy",
    ]
    assert service.current(session) == CUSTOM_PRESET


def test_pin_stored_auto_requires_integration(harness: Any) -> None:
    service: PermissionPresetService = harness.permissionPresets
    session = harness.sessions.create("stored-auto")
    session.append(EventType.PERMISSION_PRESET, {"preset": AUTO_PRESET})
    with pytest.raises(PermissionPresetError, match="without its active integration"):
        service.pin_initial_permission(session)
    admitted: list[str] = []
    dispose = service.register_auto(lambda: admitted.append("yes"))
    service.pin_initial_permission(session)
    assert admitted == ["yes"]
    assert _types(session) == [
        "permission/preset",
        "sandbox/mode",
        "approval/policy",
    ]
    dispose()


# -- service: command ------------------------------------------------------


@pytest.mark.asyncio
async def test_permission_command(harness: Any) -> None:
    await harness.flush()
    session = harness.sessions.create("command")
    agent = _agent(session)
    execution = await harness.commands.execute(agent, "/permission")
    assert execution is not None
    assert execution.result.kind == "success"
    assert execution.result.text == (
        "current preset workspace-write "
        "(available: read-only, workspace-write, danger-full-access)"
    )
    execution = await harness.commands.execute(agent, "/permission missing")
    assert execution is not None
    assert execution.result.kind == "error"
    assert execution.result.text == (
        'unknown preset "missing" '
        "(available: read-only, workspace-write, danger-full-access)"
    )
    execution = await harness.commands.execute(agent, "/permission read-only")
    assert execution is not None
    assert execution.result.kind == "success"
    assert execution.result.text == "preset read-only"
    assert harness.permissionPresets.current(session) == "read-only"


@pytest.mark.asyncio
async def test_agent_created_listener_pins_new_session(harness: Any) -> None:
    await harness.flush()
    session = harness.sessions.create("created")
    harness.emit("agent/created", {"agent": _agent(session)})
    assert _types(session) == [
        "permission/preset",
        "sandbox/mode",
        "approval/policy",
    ]
    harness.emit("agent/created", {"agent": None})
    assert len(session.events) == 3


@pytest.mark.asyncio
async def test_init_without_projection_registry() -> None:
    ctx = Context()
    ctx.provide("shell", SimpleNamespace(sandbox_mode="workspace-write"))
    ctx.provide("approval", SimpleNamespace(policy="ask"))
    ctx.provide("sessionProjections", None)
    service = PermissionPresetService(ctx)
    with pytest.raises(RuntimeError, match="sessionProjections"):
        await service.init()


class _Session:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))


class _Handle:
    def __init__(self) -> None:
        self.disposed = False

    def state_of(self, session: Any) -> dict[str, Any]:
        return _state()

    def dispose(self) -> None:
        self.disposed = True


class _Registry:
    def __init__(self) -> None:
        self.definition: Any = None

    def register(self, definition: Any) -> _Handle:
        self.definition = definition
        return _Handle()


@pytest.mark.asyncio
async def test_init_without_commands() -> None:
    ctx = Context()
    shell = SimpleNamespace(sandbox_mode="workspace-write")
    approval = SimpleNamespace(
        policy="ask",
        set_session_policy=lambda session, policy: session.append(
            EventType.APPROVAL_POLICY, {"policy": policy}
        ),
    )
    existing = _Session()
    created = _Session()
    ctx.provide("shell", shell)
    ctx.provide("approval", approval)
    ctx.provide("sessions", SimpleNamespace(get=lambda session_id: None))
    ctx.provide("agents", SimpleNamespace(list=lambda: [_agent(existing)]))
    registry = _Registry()
    ctx.provide("sessionProjections", registry)
    ctx.provide(
        "sandboxPolicy",
        SimpleNamespace(
            set_mode=lambda session, mode: session.append(
                EventType.SANDBOX_MODE, {"mode": mode}
            )
        ),
    )
    service = PermissionPresetService(ctx)
    dispose = await service.init()
    definition = registry.definition
    assert definition.key == "permissions"
    assert definition.state_version == 2
    assert definition.wire.view(definition.init({}, 0)) == {
        "currentValue": "workspace-write"
    }
    assert [event_type for event_type, _ in existing.events] == [
        "permission/preset",
        "sandbox/mode",
        "approval/policy",
    ]
    ctx.emit("agent/created", {"agent": _agent(created)})
    assert [event_type for event_type, _ in created.events] == [
        "permission/preset",
        "sandbox/mode",
        "approval/policy",
    ]
    ctx.emit("agent/created", {"agent": None})
    assert len(created.events) == 3
    dispose()
    dispose()
