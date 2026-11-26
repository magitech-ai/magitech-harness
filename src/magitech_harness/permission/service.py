"""User-facing permission presets over sandbox-mode and approval-policy knobs.

Ports ``packages/interaction/permission-presets/src/index.ts`` from the pinned
dsh source. A preset switch records ``permission/preset`` as durable user
intent, then writes each changed knob through its canonical setter. Execution,
prompt narration, and replay keep reading the knob folds. The ``permissions``
Session projection exposes the effective current value; the process catalog
lists every selectable preset. The write side is the ``/permission`` command.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, cast

from animas import Context, Service

from magitech_harness.approval.service import (
    ALL_POLICIES,
    ApprovalPolicy,
)
from magitech_harness.commands.types import (
    CommandDefinition,
    CommandInputDescriptor,
    CommandResult,
)
from magitech_harness.llm.error import HarnessError
from magitech_harness.sandbox.types import SANDBOX_MODES, SandboxMode
from magitech_harness.session.events import EventType
from magitech_harness.session.projection_registry import (
    ProjectionDefinition,
    ProjectionHandle,
    WireView,
)

#: Derived current value when the effective knobs match no available preset.
CUSTOM_PRESET = "custom"
#: Reserved identity of the experimental per-call review preset.
AUTO_PRESET = "auto"
RESERVED_PRESETS = frozenset({CUSTOM_PRESET, AUTO_PRESET})

#: The schema default table: the two bundles dsh ships without a base config.
DEFAULT_PRESETS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "workspace-write": MappingProxyType(
            {
                "sandbox": "workspace-write",
                "approval": "ask",
                "name": "workspace-write",
                "description": (
                    "Write inside the workspace and permitted temporary "
                    "directories; wider retries require approval."
                ),
            }
        ),
        "danger-full-access": MappingProxyType(
            {
                "sandbox": "danger-full-access",
                "approval": "never",
                "name": "danger-full-access",
                "description": "Full file access without approval prompts.",
            }
        ),
    }
)

#: Fixed execution bundle for the live Auto integration.
AUTO_PRESET_SPEC = ("danger-full-access", "ask")

_EMPTY_KNOBS: dict[str, Any] = {
    "preset": None,
    "sandbox": None,
    "approval": None,
}

_PROJECTION_FIELDS = frozenset({"preset", "sandbox", "approval", "seeded"})
_SELECTION_FIELDS = frozenset({"currentValue"})


class PermissionPresetError(HarnessError):
    """Stable failure taxonomy for the permission-preset domain."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message, code)


@dataclass(frozen=True)
class PresetSpec:
    """One preset's sandbox/approval bundle and optional presentation."""

    sandbox: SandboxMode
    approval: ApprovalPolicy
    name: str | None = None
    description: str | None = None


def _require_name(value: Any, subject: str) -> str:
    if not isinstance(value, str) or not value:
        raise PermissionPresetError(
            f"permission: {subject} must be a non-empty string",
            "PERMISSION_INVALID_CONFIG",
        )
    return value


def _parse_presets(value: Any) -> dict[str, PresetSpec]:
    """Validate the configured preset table into immutable specs."""
    if not isinstance(value, Mapping):
        raise PermissionPresetError(
            "permission: presets must be a mapping of name to bundle",
            "PERMISSION_INVALID_CONFIG",
        )
    presets: dict[str, PresetSpec] = {}
    for name, raw in value.items():
        key = _require_name(name, "preset names")
        if key in RESERVED_PRESETS:
            raise PermissionPresetError(
                f'permission: "{key}" is reserved and cannot name a configured preset',
                "PERMISSION_INVALID_CONFIG",
            )
        if not isinstance(raw, Mapping):
            raise PermissionPresetError(
                f'permission: preset "{key}" must be a mapping',
                "PERMISSION_INVALID_CONFIG",
            )
        unknown = set(raw) - {"sandbox", "approval", "name", "description"}
        if unknown:
            raise PermissionPresetError(
                f'permission: preset "{key}" has unknown fields '
                f"{', '.join(sorted(unknown))}",
                "PERMISSION_INVALID_CONFIG",
            )
        sandbox = raw.get("sandbox")
        if sandbox not in SANDBOX_MODES:
            raise PermissionPresetError(
                f'permission: preset "{key}" sandbox must be one of '
                f"{', '.join(SANDBOX_MODES)}",
                "PERMISSION_INVALID_CONFIG",
            )
        approval = raw.get("approval")
        if approval not in ALL_POLICIES:
            raise PermissionPresetError(
                f'permission: preset "{key}" approval must be ask|never',
                "PERMISSION_INVALID_CONFIG",
            )
        label = raw.get("name")
        if label is not None and (not isinstance(label, str) or not label):
            raise PermissionPresetError(
                f'permission: preset "{key}" name must be a non-empty string',
                "PERMISSION_INVALID_CONFIG",
            )
        description = raw.get("description")
        if description is not None and not isinstance(description, str):
            raise PermissionPresetError(
                f'permission: preset "{key}" description must be a string',
                "PERMISSION_INVALID_CONFIG",
            )
        presets[key] = PresetSpec(
            sandbox=cast(SandboxMode, sandbox),
            approval=cast(ApprovalPolicy, approval),
            name=label,
            description=description,
        )
    return presets


def _validate_state(value: Any) -> dict[str, Any]:
    """Validate one persisted ``permissions`` projection cell."""
    if not isinstance(value, dict):
        raise TypeError("permissions projection must be an object")
    if set(value) != _PROJECTION_FIELDS:
        raise ValueError("permissions projection fields are invalid")
    preset = value["preset"]
    if preset is not None and (not isinstance(preset, str) or not preset):
        raise ValueError("permissions.preset must be a non-empty string or null")
    sandbox = value["sandbox"]
    if sandbox is not None and sandbox not in SANDBOX_MODES:
        raise ValueError("permissions.sandbox must be a valid sandbox mode or null")
    approval = value["approval"]
    if approval is not None and approval not in ALL_POLICIES:
        raise ValueError("permissions.approval must be ask|never or null")
    seeded = value["seeded"]
    if not isinstance(seeded, bool):
        raise TypeError("permissions.seeded must be a boolean")
    return {
        "preset": preset,
        "sandbox": sandbox,
        "approval": approval,
        "seeded": seeded,
    }


def _init_state(header: dict[str, Any], inherited_event_count: int) -> dict[str, Any]:
    """State for the empty log: every knob at its composition default."""
    return {**_EMPTY_KNOBS, "seeded": False}


def _apply_event(state: dict[str, Any], event: Any) -> dict[str, Any]:
    """One-event transition; unrelated events keep the same reference."""
    if event.type == EventType.PERMISSION_PRESET:
        preset = event.payload.get("preset")
        if not isinstance(preset, str) or not preset:
            raise ValueError("permission/preset payload must carry a non-empty preset")
        if preset == state["preset"]:
            return state
        return {**state, "preset": preset}
    if event.type == EventType.SANDBOX_MODE:
        sandbox = event.payload.get("mode")
        if sandbox not in SANDBOX_MODES:
            raise ValueError("sandbox/mode payload must carry a valid mode")
        if sandbox == state["sandbox"]:
            return state
        return {**state, "sandbox": sandbox}
    if event.type == EventType.APPROVAL_POLICY:
        approval = event.payload.get("policy")
        if approval not in ALL_POLICIES:
            raise ValueError("approval/policy payload must carry ask|never")
        if approval == state["approval"]:
            return state
        return {**state, "approval": approval}
    if event.type == EventType.SESSION_END_SEED:
        if state["seeded"]:
            return state
        return {**state, "seeded": True}
    return state


def _validate_selection(value: Any) -> dict[str, str]:
    """Validate one client-facing ``permissions`` wire view."""
    if not isinstance(value, dict) or set(value) != _SELECTION_FIELDS:
        raise ValueError("permissions view must be an object with currentValue")
    current = value["currentValue"]
    if not isinstance(current, str) or not current:
        raise ValueError("permissions.currentValue must be a non-empty string")
    return {"currentValue": current}


class PermissionPresetService(Service):
    """Configured presets, the fixed Auto hook, and their write paths."""

    inject: ClassVar[list[str]] = [
        "shell",
        "approval",
        "sessions",
        "agents",
        "sessionProjections",
    ]

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, "permissionPresets")
        cfg = dict(config or {})
        unknown = set(cfg) - {"presets", "defaultPreset"}
        if unknown:
            raise PermissionPresetError(
                "permission: unknown config fields " + ", ".join(sorted(unknown)),
                "PERMISSION_INVALID_CONFIG",
            )
        table = cfg.get("presets")
        self._presets = _parse_presets(DEFAULT_PRESETS if table is None else table)
        shell = ctx.get("shell", None, strict=True)
        if getattr(shell, "sandbox_mode", None) is None:
            raise PermissionPresetError(
                "permission: the mounted bash executor does not confine "
                "(no sandboxMode) - presets bundle a sandbox mode, so "
                "composing this plugin over an unconfined executor is a "
                "misconfiguration",
                "PERMISSION_INVALID_CONFIG",
            )
        self._auto_admit: Callable[[], None] | None = None
        self._inferred_default = self._derive(_EMPTY_KNOBS)
        configured = cfg.get("defaultPreset")
        if configured is None:
            default = self._inferred_default
        else:
            default = _require_name(configured, "defaultPreset")
        if default == CUSTOM_PRESET:
            raise PermissionPresetError(
                "permission: composed sandbox and approval defaults match no "
                "preset; configure defaultPreset explicitly",
                "PERMISSION_INVALID_CONFIG",
            )
        self.resolve(default)
        self._default_preset = default
        self._handle: ProjectionHandle[dict[str, Any]] | None = None

    # -- lifecycle ---------------------------------------------------------

    async def init(self) -> Callable[[], None]:
        """Register the ``permissions`` projection and its write paths."""
        registry = self.ctx.get("sessionProjections", None, strict=True)
        if registry is None:
            raise RuntimeError(
                "permissionPresets requires the sessionProjections service"
            )
        definition = ProjectionDefinition(
            key="permissions",
            state_schema=_validate_state,
            init=_init_state,
            apply=_apply_event,
            wire=WireView(
                view_schema=_validate_selection,
                view=lambda state: {"currentValue": self._derive(state)},
                summary=False,
            ),
            state_version=2,
        )
        self._handle = registry.register(definition)

        def on_created(payload: Mapping[str, Any]) -> None:
            agent = payload.get("agent")
            if agent is not None:
                self.pin_initial_permission(agent.session)

        dispose_created = self.ctx.on("agent/created", on_created, prepend=True)

        agents = self.ctx.get("agents", None, strict=True)
        for agent in agents.list():
            self.pin_initial_permission(agent.session)

        dispose_command = self._register_command()

        def dispose() -> None:
            dispose_command()
            dispose_created()
            if self._handle is not None:
                self._handle.dispose()
                self._handle = None

        return dispose

    # -- catalog -----------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Every switchable preset name, in contribution order."""
        names = list(self._presets)
        if self._auto_admit is not None:
            names.append(AUTO_PRESET)
        return names

    @property
    def default_preset(self) -> str:
        """The preset pinned into a new session."""
        return self._default_preset

    def catalog(self) -> dict[str, Any]:
        """The complete process-level catalog exposed to current-session UI."""
        return {
            "options": [self.option_of(name) for name in self.names],
            "defaultOptions": [self.option_of(name) for name in self._presets],
            "defaultPreset": self.default_preset,
        }

    def option_of(self, name: str) -> dict[str, Any]:
        """Build the client option for a preset or the derived custom value."""
        if name == CUSTOM_PRESET:
            return {
                "value": CUSTOM_PRESET,
                "name": "Custom",
                "description": (
                    "Current sandbox and approval settings do not match a preset."
                ),
            }
        spec = self.resolve(name)
        option: dict[str, Any] = {"value": name, "name": spec.name or name}
        if spec.description is not None:
            option["description"] = spec.description
        return option

    def resolve(self, name: str) -> PresetSpec:
        """Resolve an available preset's knob bundle."""
        spec = self._spec_of(name)
        if spec is None:
            raise PermissionPresetError(
                f'permission: unknown preset "{name}" (known: {", ".join(self.names)})',
                "PERMISSION_UNKNOWN_PRESET",
            )
        return spec

    def _spec_of(self, name: str) -> PresetSpec | None:
        spec = self._presets.get(name)
        if spec is not None:
            return spec
        if name == AUTO_PRESET and self._auto_admit is not None:
            sandbox, approval = AUTO_PRESET_SPEC
            return PresetSpec(
                sandbox=cast(SandboxMode, sandbox),
                approval=cast(ApprovalPolicy, approval),
            )
        return None

    # -- current selection -------------------------------------------------

    def current(self, session: Any) -> str:
        """The effective preset for *session*, or ``custom`` when none matches."""
        return self._derive(self._state_of(session))

    def _state_of(self, session: Any) -> dict[str, Any]:
        if self._handle is None:
            raise PermissionPresetError(
                "permission: presets are not ready; await plugin activation "
                "before creating or restoring a session",
                "PERMISSION_NOT_READY",
            )
        return self._handle.state_of(session)

    def _derive(self, state: Mapping[str, Any]) -> str:
        sandbox = (
            state["sandbox"]
            if state["sandbox"] is not None
            else self._composed_sandbox()
        )
        approval = (
            state["approval"]
            if state["approval"] is not None
            else self._composed_approval()
        )
        preset = cast(str | None, state["preset"])
        if preset is not None:
            spec = self._spec_of(preset)
            if (
                spec is not None
                and spec.sandbox == sandbox
                and spec.approval == approval
            ):
                return preset
            if (
                preset == AUTO_PRESET
                and spec is not None
                and spec.sandbox == sandbox
                and approval == "never"
            ):
                return AUTO_PRESET
        for name, candidate in self._presets.items():
            if candidate.sandbox == sandbox and candidate.approval == approval:
                return name
        return CUSTOM_PRESET

    def _composed_sandbox(self) -> SandboxMode:
        shell = self.ctx.get("shell", None, strict=True)
        mode = getattr(shell, "sandbox_mode", None)
        if mode is None:
            raise PermissionPresetError(
                "permission: the mounted bash executor does not confine "
                "(no sandboxMode)",
                "PERMISSION_INVALID_CONFIG",
            )
        return cast(SandboxMode, mode)

    def _composed_approval(self) -> ApprovalPolicy:
        approval = self.ctx.get("approval", None, strict=True)
        policy = getattr(approval, "policy", None)
        if policy not in ALL_POLICIES:
            raise PermissionPresetError(
                "permission: approval service has no valid default policy",
                "PERMISSION_INVALID_CONFIG",
            )
        return cast(ApprovalPolicy, policy)

    # -- writes ------------------------------------------------------------

    def set(self, session: Any, name: str) -> None:
        """Switch one session to a preset through its durable identity."""
        self.apply(
            session,
            name,
            lambda policy: self._set_session_approval(session, policy),
        )

    def set_preset(self, session_id: str, preset: str) -> dict[str, str]:
        """Switch one live session named by id; returns its current value."""
        sessions = self.ctx.get("sessions", None, strict=True)
        session = sessions.get(session_id)
        if session is None:
            raise PermissionPresetError(
                f'permission: no live session "{session_id}"',
                "PERMISSION_UNKNOWN_SESSION",
            )
        self.set(session, preset)
        return {"currentValue": self.current(session)}

    def apply(
        self,
        session: Any,
        name: str,
        set_approval: Callable[[ApprovalPolicy], None],
    ) -> None:
        """Record a changed preset, then update each changed knob."""
        spec = self.resolve(name)
        if name == AUTO_PRESET and self._auto_admit is not None:
            self._auto_admit()
        current = self.current(session)
        knobs = self._state_of(session)
        if current != name:
            session.append(EventType.PERMISSION_PRESET, {"preset": name})
        if spec.sandbox != (
            knobs["sandbox"]
            if knobs["sandbox"] is not None
            else self._composed_sandbox()
        ):
            sandbox_policy = self.ctx.get("sandboxPolicy", None, strict=True)
            sandbox_policy.set_mode(session, spec.sandbox)
        if spec.approval != (
            knobs["approval"]
            if knobs["approval"] is not None
            else self._composed_approval()
        ):
            set_approval(spec.approval)

    def _set_session_approval(self, session: Any, policy: ApprovalPolicy) -> None:
        session.append(EventType.APPROVAL_POLICY, {"policy": policy})

    def pin_initial_permission(self, session: Any) -> None:
        """Fill every missing permission fact during Agent creation."""
        state = self._state_of(session)
        preset = state["preset"]
        sandbox = state["sandbox"]
        approval = state["approval"]
        seeded = state["seeded"]
        if preset == AUTO_PRESET:
            if self._auto_admit is None:
                raise PermissionPresetError(
                    'permission: cannot restore preset "auto" without its '
                    "active integration",
                    "PERMISSION_UNKNOWN_PRESET",
                )
            self._auto_admit()
        if preset is None and sandbox is None and approval is None and not seeded:
            name = self.default_preset
            spec = self.resolve(name)
            session.append(EventType.PERMISSION_PRESET, {"preset": name})
            sandbox_policy = self.ctx.get("sandboxPolicy", None, strict=True)
            sandbox_policy.set_mode(session, spec.sandbox)
            self._set_session_approval(session, spec.approval)
            return
        effective = self._derive(state)
        if preset is None and effective != CUSTOM_PRESET:
            session.append(EventType.PERMISSION_PRESET, {"preset": effective})
        if sandbox is None:
            sandbox_policy = self.ctx.get("sandboxPolicy", None, strict=True)
            sandbox_policy.set_mode(session, self._composed_sandbox())
        if approval is None:
            self._set_session_approval(session, self._composed_approval())

    # -- auto integration --------------------------------------------------

    def register_auto(self, admit: Callable[[], None]) -> Callable[[], Any]:
        """Publish the fixed current-session Auto preset for one integration."""
        if self._auto_admit is not None:
            raise PermissionPresetError(
                'permission: preset "auto" is already registered',
                "PERMISSION_AUTO_CONFLICT",
            )

        def setup() -> Callable[[], None]:
            self._auto_admit = admit
            self._emit_catalog_changed()

            def dispose() -> None:
                self._auto_admit = None
                self._emit_catalog_changed()

            return dispose

        return cast(
            Callable[[], Any],
            self.ctx.effect(setup, "permissionPresets.registerAuto()"),
        )

    def _emit_catalog_changed(self) -> None:
        """Dispatch to every observer independently so none can veto the change."""
        event = "permission-presets/catalog-changed"
        listeners = list(self.ctx.events._listeners.get(event, []))
        for listener in listeners:
            try:
                returned = listener.callback()
                if inspect.isawaitable(returned):
                    task = asyncio.ensure_future(returned)

                    def observe(future: Any) -> None:
                        if future.cancelled():
                            return
                        error = future.exception()
                        if error is not None:
                            self.ctx.logger.warning(
                                f"permission: catalog-changed listener failed: {error}"
                            )

                    task.add_done_callback(observe)
            except Exception as error:  # noqa: BLE001 -- isolation from dsh
                self.ctx.logger.warning(
                    f"permission: catalog-changed listener failed: {error}"
                )

    # -- command -----------------------------------------------------------

    def _register_command(self) -> Callable[[], None]:
        commands = self.ctx.get("commands", None, strict=False)
        if commands is None:
            return lambda: None

        def handler(invocation: Any) -> CommandResult:
            name = invocation.raw_input.strip()
            names = ", ".join(self.names)
            if name == "":
                return CommandResult.success(
                    f"current preset {self.current(invocation.agent.session)} "
                    f"(available: {names})"
                )
            if name not in self.names:
                return CommandResult.error(
                    f'unknown preset "{name}" (available: {names})'
                )
            self.apply(
                invocation.agent.session,
                name,
                lambda policy: self._set_session_approval(
                    invocation.agent.session, policy
                ),
            )
            return CommandResult.success(f"preset {name}")

        return cast(
            Callable[[], None],
            commands.register(
                CommandDefinition(
                    name="permission",
                    description=(
                        "Switch the permission preset (sandbox mode + approval policy)"
                    ),
                    handler=handler,
                    definition_id="@deepseek-ai/dsh-permission-presets",
                    input=CommandInputDescriptor(hint="<preset>"),
                )
            ),
        )


def apply(
    ctx: Context, config: Mapping[str, Any] | None = None
) -> PermissionPresetService:
    """Mount ``ctx.permissionPresets``."""
    return PermissionPresetService(ctx, config)


apply.inject = [  # type: ignore[attr-defined]
    "shell",
    "approval",
    "sessions",
    "agents",
    "sessionProjections",
]


__all__ = [
    "AUTO_PRESET",
    "AUTO_PRESET_SPEC",
    "CUSTOM_PRESET",
    "DEFAULT_PRESETS",
    "RESERVED_PRESETS",
    "PermissionPresetError",
    "PermissionPresetService",
    "PresetSpec",
    "apply",
]
