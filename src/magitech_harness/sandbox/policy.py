"""The single sandbox policy home: modes, roots and session overrides."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from animas import Context, Service

from magitech_harness.sandbox.types import (
    SANDBOX_MODES,
    SandboxError,
    SandboxInvalidRootError,
)

POLICY_CONTEXT = (
    "Current DSH file policy: {mode}. {detail} Do not refuse a required modification "
    "from this policy alone: try an available tool normally and follow any denial "
    "and escalation guidance it returns."
)


def _execution_world_path(path: Any) -> bool:
    return isinstance(path, str) and os.path.isabs(path) and "\0" not in path


def _resolve_roots(roots: Any, subject: str) -> tuple[str, ...] | None:
    if roots is None:
        return None
    if not isinstance(roots, (list, tuple)) or not roots:
        return None
    for root in roots:
        if not _execution_world_path(root):
            raise SandboxError(
                "sandbox-policy: writable directories must be absolute execution-world paths without NUL bytes",
                "SANDBOX_INVALID_CONFIG",
            )
    return tuple(dict.fromkeys(str(root) for root in roots))


class SandboxPolicyService(Service):
    """``ctx.sandboxPolicy``: resolve one file-effect policy per call."""

    inject: ClassVar[list[str]] = ["sessionProjections"]

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, "sandboxPolicy")
        cfg = dict(config or {})
        mode = cfg.get("mode", "read-only")
        if mode not in SANDBOX_MODES:
            raise SandboxError("sandbox-policy: invalid mode", "SANDBOX_INVALID_CONFIG")
        root = cfg.get("workspaceRoot") or os.getcwd()
        if not _execution_world_path(root):
            raise SandboxInvalidRootError(
                str(root),
                "the workspace root must be an absolute execution-world path without NUL bytes",
                "sandbox workspace root",
            )
        self.default_mode = mode
        self.workspace_root = str(root)
        self.configured_roots = _resolve_roots(
            cfg.get("additionalWritableRoots"), "additional writable directory"
        )
        self._registrations: dict[int, dict[str, str]] = {}
        self._handle: Any = None

    async def init(self) -> Callable[[], None]:
        """Register the ``sandboxMode`` session projection and policy context."""
        from magitech_harness.session.projection_registry import (
            ProjectionDefinition,
            WireView,
        )

        registry = self.ctx.get("sessionProjections", None, strict=True)
        if registry is None:
            raise RuntimeError("sandboxPolicy requires sessionProjections")
        definition = ProjectionDefinition(
            key="sandboxMode",
            state_schema=self._validate_state,
            init=lambda header, inherited: None,
            apply=self._apply_event,
            wire=WireView(
                view_schema=self._validate_state,
                view=lambda state: state,
                summary=False,
            ),
            state_version=1,
        )
        self._handle = registry.register(definition)
        system_prompt = self.ctx.get("systemPrompt", None, strict=False)
        if system_prompt is not None:
            system_prompt.context(
                {
                    "name": "sandbox:policy",
                    "order": system_prompt.get_context_order("SANDBOX_POLICY")
                    if hasattr(system_prompt, "get_context_order")
                    else 110,
                    "text": lambda ctx_arg: self._render_context(ctx_arg),
                }
            )
        dispose: Callable[[], None] = self._handle.dispose
        return dispose

    @staticmethod
    def _validate_state(value: Any) -> Any:
        if value is None:
            return None
        if value not in SANDBOX_MODES:
            raise TypeError("sandbox mode projection must be a valid mode or null")
        return value

    @staticmethod
    def _apply_event(state: Any, event: Any) -> Any:
        if event.type == "sandbox/mode":
            return event.payload.get("mode", state)
        return state

    def _override(self, session: Any) -> str | None:
        if self._handle is None or session is None:
            return None
        override: str | None = self._handle.state_of(session)
        return override

    def _session_root(self, session: Any) -> str:
        if session is None:
            return self.workspace_root
        working_directory = self.ctx.get("workingDirectory", None, strict=False)
        get = getattr(working_directory, "get", None)
        root = get(session) if callable(get) else None
        if root is None:
            header = getattr(session, "header", {}) or {}
            root = header.get("cwd")
        if root is None:
            return self.workspace_root
        if not _execution_world_path(root):
            raise SandboxInvalidRootError(
                str(root),
                "the Session working directory must be an absolute execution-world path without NUL bytes",
                "sandbox workspace root",
            )
        return str(root)

    def resolve(self, session: Any = None, mode: str | None = None) -> dict[str, Any]:
        """Resolve the complete per-call policy (explicit > session > default)."""
        effective = mode or self._override(session) or self.default_mode
        roots = list(self.configured_roots or ())
        registrations = (
            self._registrations.get(id(session), {}) if session is not None else {}
        )
        roots.extend(registrations.values())
        resolved_roots = _resolve_roots(roots, "additional writable directory")
        policy: dict[str, Any] = {
            "mode": effective,
            "workspaceRoot": self._session_root(session),
        }
        if resolved_roots:
            policy["additionalWritableRoots"] = resolved_roots
        if session is not None:
            policy["sessionId"] = getattr(session, "session_id", None)
        return policy

    def register_writable_root(
        self, session: Any, directory: str
    ) -> Callable[[], None]:
        """Register an extra writable root for one live session."""
        if not _execution_world_path(directory):
            raise SandboxError(
                "sandbox-policy: writable directories must be absolute execution-world paths without NUL bytes",
                "SANDBOX_INVALID_CONFIG",
            )
        registrations = self._registrations.setdefault(id(session), {})
        token = str(len(registrations))
        registrations[token] = directory

        def dispose() -> None:
            registrations.pop(token, None)

        return dispose

    def set_mode(self, session: Any, mode: str) -> None:
        """Append one log-only ``sandbox/mode`` event."""
        if mode not in SANDBOX_MODES:
            raise SandboxError("sandbox-policy: invalid mode", "SANDBOX_INVALID_CONFIG")
        session.append("sandbox/mode", {"mode": mode})

    def _render_context(self, ctx_arg: Mapping[str, Any]) -> str:
        session = ctx_arg.get("session")
        policy = self.resolve(session)
        mode = policy["mode"]
        detail = {
            "read-only": (
                "Any available operation enforced by the DSH file sandbox cannot "
                "modify files in the standing mode."
            ),
            "workspace-write": (
                "Any available operation enforced by the DSH file sandbox may "
                f"modify files under the session workspace: {policy['workspaceRoot']!r}."
            ),
            "danger-full-access": (
                "The DSH file sandbox does not restrict file modifications by "
                "available operations."
            ),
        }[mode]
        return POLICY_CONTEXT.format(mode=mode, detail=detail)


def apply(
    ctx: Context, config: Mapping[str, Any] | None = None
) -> SandboxPolicyService:
    """Mount ``ctx.sandboxPolicy``."""
    return SandboxPolicyService(ctx, config)


apply.inject = ["sessionProjections"]  # type: ignore[attr-defined]

__all__ = ["POLICY_CONTEXT", "SandboxPolicyService", "apply"]
