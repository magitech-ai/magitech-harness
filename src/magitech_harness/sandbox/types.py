"""Vocabulary for the same-world process confinement seam."""

from __future__ import annotations

from typing import Literal

from magitech_harness.llm.error import HarnessError

SandboxMode = Literal["read-only", "workspace-write", "danger-full-access"]
ConfinedSandboxMode = Literal["read-only", "workspace-write"]
SandboxEnforcement = Literal["full", "partial"]
SANDBOX_MODES = ("read-only", "workspace-write", "danger-full-access")
SANDBOX_UNAVAILABLE = "SANDBOX_UNAVAILABLE"
SANDBOX_INVALID_ROOT = "SANDBOX_INVALID_ROOT"
SANDBOX_PREPARATION_FAILED = "SANDBOX_PREPARATION_FAILED"
SANDBOX_INVALID_TEMP_ROOT = "SANDBOX_INVALID_TEMP_ROOT"
SANDBOX_PREPARATION_ERROR_CODES = (
    SANDBOX_PREPARATION_FAILED,
    SANDBOX_INVALID_TEMP_ROOT,
)


class SandboxError(HarnessError):
    """Base class for sandbox failures."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message, code)


class SandboxUnavailableError(SandboxError):
    """Fail-closed refusal when no confinement backend is usable."""

    def __init__(self, mode: str, detail: str | None = None) -> None:
        message = (
            f'sandbox mode "{mode}" is requested but no sandbox backend is usable '
            "on this host; refusing to run the command unconfined. Install "
            "bubblewrap or run a Landlock-enforcing kernel (Linux), ensure "
            "sandbox-exec is usable (macOS), or ensure the ACL restricted-token "
            "runner can start (Windows) — otherwise switch the consumer to "
            "danger-full-access."
        )
        if detail:
            message += f" Runner failure: {detail}"
        super().__init__(message, SANDBOX_UNAVAILABLE)


class SandboxInvalidRootError(SandboxError):
    """An additional writable root failed host-side validation."""

    def __init__(
        self,
        path: str,
        detail: str,
        subject: str = "additional writable directory",
    ) -> None:
        import json

        super().__init__(
            f"Invalid {subject} {json.dumps(path)}: {detail}", SANDBOX_INVALID_ROOT
        )


class SandboxPreparationError(SandboxError):
    """A confinement runner could not be prepared."""

    def __init__(self, message: str, code: str = SANDBOX_PREPARATION_FAILED) -> None:
        super().__init__(message, code)


class RunnerFailureRule:
    """Rule identifying a fatal runner diagnostic on stderr."""

    def __init__(
        self,
        fatal_signatures: tuple[str, ...],
        allowed_exit_codes: tuple[int, ...] | None = None,
        informational_lines: tuple[str, ...] = (),
    ) -> None:
        self.fatal_signatures = fatal_signatures
        self.allowed_exit_codes = allowed_exit_codes
        self.informational_lines = informational_lines


class ConfinedArgv:
    """The exact argv to spawn, plus the backend's enforcement facts."""

    def __init__(
        self,
        argv: list[str],
        enforcement: SandboxEnforcement,
        denial_signatures: tuple[str, ...] = (),
        runner_failure_rules: tuple[RunnerFailureRule, ...] = (),
    ) -> None:
        self.argv = argv
        self.enforcement = enforcement
        self.denial_signatures = denial_signatures
        self.runner_failure_rules = runner_failure_rules


__all__ = [
    "SANDBOX_INVALID_ROOT",
    "SANDBOX_INVALID_TEMP_ROOT",
    "SANDBOX_MODES",
    "SANDBOX_PREPARATION_ERROR_CODES",
    "SANDBOX_PREPARATION_FAILED",
    "SANDBOX_UNAVAILABLE",
    "ConfinedArgv",
    "ConfinedSandboxMode",
    "RunnerFailureRule",
    "SandboxEnforcement",
    "SandboxError",
    "SandboxInvalidRootError",
    "SandboxMode",
    "SandboxPreparationError",
    "SandboxUnavailableError",
]
