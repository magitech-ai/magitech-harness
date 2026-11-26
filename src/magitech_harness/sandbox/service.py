"""The process-confinement seam, backed by bubblewrap on Linux.

Ports ``packages/sandbox/sandbox/src`` and ``sandbox-local``. ``confine`` is
fail-closed: it returns enforcing argv or raises ``SANDBOX_UNAVAILABLE``.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from animas import Context, Service

from magitech_harness.sandbox.roots import workspace_writable_roots
from magitech_harness.sandbox.types import (
    ConfinedArgv,
    RunnerFailureRule,
    SandboxError,
    SandboxUnavailableError,
)

DENIAL_SIGNATURES: dict[str, tuple[str, ...]] = {
    "bwrap": ("read-only file system",),
    "landlock": ("permission denied",),
    "runnerCommand": ("read-only file system", "permission denied"),
}
RUNNER_FAILURE_RULES: dict[str, tuple[RunnerFailureRule, ...]] = {
    "bwrap": (RunnerFailureRule(("bwrap: ",), informational_lines=()),),
}
STATIC_ENFORCEMENT: dict[str, str] = {
    "bwrap": "full",
    "landlock": "full",
    "windows-acl": "partial",
}


class SandboxService(Service):
    """``ctx.sandbox``: argv-level confinement with per-call policies."""

    inject: ClassVar[list[str]] = []

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, "sandbox")
        cfg = dict(config or {})
        runner = cfg.get("runnerCommand") or []
        signatures = cfg.get("runnerFailureSignatures") or []
        if signatures and not runner:
            raise SandboxError(
                "sandbox-local: runnerFailureSignatures requires runnerCommand",
                "SANDBOX_INVALID_CONFIG",
            )
        if runner and not signatures:
            raise SandboxError(
                "sandbox-local: runnerCommand requires at least one runnerFailureSignatures entry",
                "SANDBOX_INVALID_CONFIG",
            )
        for signature in signatures:
            if (
                not isinstance(signature, str)
                or not signature.strip()
                or "\n" in signature
                or "\r" in signature
            ):
                raise SandboxError(
                    "sandbox-local: runnerFailureSignatures entries must be non-empty single-line strings",
                    "SANDBOX_INVALID_CONFIG",
                )
        self.runner_command = list(runner)
        self.runner_signatures = tuple(signatures)
        self.bwrap_path = cfg.get("bwrapPath") or shutil.which("bwrap")
        self.probe_timeout_ms = cfg.get("probeTimeoutMs", 5000)
        if (
            isinstance(self.probe_timeout_ms, bool)
            or not isinstance(self.probe_timeout_ms, (int, float))
            or not math.isfinite(self.probe_timeout_ms)
            or self.probe_timeout_ms <= 0
        ):
            raise SandboxError(
                "sandbox-local: probeTimeoutMs must be a positive finite number",
                "SANDBOX_INVALID_CONFIG",
            )
        self._bwrap_verdict: bool | None = None

    def _probe_bwrap(self) -> bool:
        """Run the read-only profile once (bounded by ``probeTimeoutMs``)."""
        if self._bwrap_verdict is None:
            if self.bwrap_path is None:
                self._bwrap_verdict = False
            else:
                profile = self._profile_argv(
                    {"mode": "read-only", "workspaceRoot": "/"}
                )
                try:
                    result = subprocess.run(
                        [*profile, "--", "true"],
                        timeout=self.probe_timeout_ms / 1000,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    self._bwrap_verdict = result.returncode == 0
                except (OSError, subprocess.TimeoutExpired):
                    self._bwrap_verdict = False
        return self._bwrap_verdict

    def _bwrap_profile_flags(self, policy: Mapping[str, Any]) -> list[str]:
        """The bwrap profile arguments, without an executable."""
        args: list[str] = [
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--unshare-pid",
            "--proc",
            "/proc",
            "--die-with-parent",
        ]
        if policy.get("mode") == "workspace-write":
            args.extend(["--tmpfs", "/tmp"])
            for root in workspace_writable_roots(policy):
                args.extend(["--bind", root, root])
        return args

    def _profile_argv(self, policy: Mapping[str, Any]) -> list[str]:
        """Build the bwrap argv for one policy."""
        assert self.bwrap_path is not None
        return [self.bwrap_path, *self._bwrap_profile_flags(policy)]

    def confine(
        self,
        argv: Sequence[str],
        policy: Mapping[str, Any],
        signal: Any = None,
    ) -> ConfinedArgv:
        """Wrap exact argv in an enforcing runner, or fail closed."""
        if signal is not None and getattr(signal, "aborted", False):
            signal.throw_if_aborted()
        mode = str(policy.get("mode", "read-only"))
        if mode == "danger-full-access":
            raise SandboxError(
                "danger-full-access is not a confined mode",
                "SANDBOX_INVALID_CONFIG",
            )
        if self.runner_command:
            runner = [*self.runner_command, *self._bwrap_profile_flags(policy)]
            signatures = DENIAL_SIGNATURES["runnerCommand"]
            rules: tuple[RunnerFailureRule, ...] = (
                RunnerFailureRule(self.runner_signatures),
            )
            enforcement = "full"
        else:
            if not self._probe_bwrap():
                raise SandboxUnavailableError(mode)
            runner = self._profile_argv(policy)
            signatures = DENIAL_SIGNATURES["bwrap"]
            rules = RUNNER_FAILURE_RULES["bwrap"]
            enforcement = STATIC_ENFORCEMENT["bwrap"]
        return ConfinedArgv(
            argv=[*runner, "--", *argv],
            enforcement=enforcement,  # type: ignore[arg-type]
            denial_signatures=signatures,
            runner_failure_rules=rules,
        )


def apply(ctx: Context, config: Mapping[str, Any] | None = None) -> SandboxService:
    """Mount ``ctx.sandbox``."""
    return SandboxService(ctx, config)


apply.inject = []  # type: ignore[attr-defined]

__all__ = [
    "DENIAL_SIGNATURES",
    "RUNNER_FAILURE_RULES",
    "STATIC_ENFORCEMENT",
    "SandboxService",
    "apply",
]
