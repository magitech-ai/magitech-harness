"""Sandbox-consuming bash executor (``ctx.shell``).

Ports ``packages/shell/bash-sandbox/src``. Confines the exact ``bash -c`` argv
through ``ctx.sandbox`` using the per-call ``ctx.sandboxPolicy`` resolution,
and stamps ``sandbox`` facts on the resulting process.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from magitech_harness.sandbox.types import (
    ConfinedArgv,
    SandboxUnavailableError,
)
from magitech_harness.shell.bash_local import LocalBashExecutor
from magitech_harness.shell.types import ShellExecSpec, ShellExecution


def matches_signature(
    exit_code: int | None, stderr: str, signatures: tuple[str, ...]
) -> bool:
    """Whether a failed process emitted a backend denial signature."""
    if exit_code is None or exit_code == 0:
        return False
    lowered = stderr.lower()
    return any(
        signature.lower() in lowered for signature in signatures if signature.strip()
    )


def classify_runner_failure(
    exit_code: int | None, stderr: str, rules: tuple[Any, ...]
) -> str | None:
    """Return the first fatal runner diagnostic, or None."""
    if exit_code is None or exit_code == 0:
        return None
    lowered_lines = [line.lower() for line in stderr.splitlines()]
    for rule in rules:
        allowed = getattr(rule, "allowed_exit_codes", None)
        if allowed is not None and exit_code not in allowed:
            continue
        informational = {
            line.strip().lower() for line in getattr(rule, "informational_lines", ())
        }
        for line in lowered_lines:
            if line.strip() in informational:
                continue
            for signature in getattr(rule, "fatal_signatures", ()):
                if signature.strip() and signature.lower() in line:
                    return line
    return None


class SandboxBashExecutor(LocalBashExecutor):
    """``ctx.shell`` provider that confines every bash invocation."""

    inject: ClassVar[list[str]] = ["subprocess", "sandbox", "sandboxPolicy"]
    _config_prefix: ClassVar[str] = "bash-sandbox"

    @property
    def sandbox_mode(self) -> str | None:
        """The deployment's standing mode (capability advertisement only)."""
        policy = self.ctx.get("sandboxPolicy", None, strict=False)
        return policy.default_mode if policy is not None else None

    async def execute(self, spec: ShellExecSpec) -> ShellExecution:
        """Execute *spec* under the resolved per-call policy."""
        policy = self.ctx.sandboxPolicy
        resolved: Mapping[str, Any] = spec.get("sandboxPolicy") or policy.resolve()
        mode = str(resolved.get("mode", "read-only"))
        if mode == "danger-full-access":
            process = await super().execute(spec)
            process.sandbox = {"mode": mode, "denied": False}
            return process

        confined: ConfinedArgv = self.ctx.sandbox.confine(
            ["bash", "-c", spec["command"]], resolved, spec.get("signal")
        )
        process = await self.execute_argv(spec, confined.argv)
        original_result = process.result

        async def result() -> Any:
            value = await original_result()
            stderr = str(value.get("stderr", {}).get("text", ""))
            runner_failure = classify_runner_failure(
                value.get("exitCode"), stderr, confined.runner_failure_rules
            )
            if runner_failure is not None:
                raise SandboxUnavailableError(mode, runner_failure)
            denied = matches_signature(
                value.get("exitCode"), stderr, confined.denial_signatures
            )
            value["sandbox"] = {
                "mode": mode,
                "denied": denied,
                "enforcement": confined.enforcement,
            }
            return value

        process.result = result  # type: ignore[assignment]
        process.sandbox = {
            "mode": mode,
            "denied": False,
            "enforcement": confined.enforcement,
        }
        return process


apply = SandboxBashExecutor
__all__ = [
    "SandboxBashExecutor",
    "apply",
    "classify_runner_failure",
    "matches_signature",
]
