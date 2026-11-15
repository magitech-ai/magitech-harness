"""Shell execution seam, parsers, and local process providers."""

from __future__ import annotations

from magitech_harness.shell.bash_local import (
    ENV_OVERRIDES as BASH_ENV_OVERRIDES,
)
from magitech_harness.shell.bash_local import (
    LocalBashExecutor,
    assert_serviceable_bash_config,
)
from magitech_harness.shell.executor import ShellExecutor
from magitech_harness.shell.pwsh_local import (
    ENCODING_PREAMBLE as PWSH_ENCODING_PREAMBLE,
)
from magitech_harness.shell.pwsh_local import (
    ENV_OVERRIDES as PWSH_ENV_OVERRIDES,
)
from magitech_harness.shell.pwsh_local import (
    LocalPwshExecutor,
    PwshLocalExecutor,
    assert_serviceable_pwsh_config,
    candidate_pwsh_paths,
    resolve_pwsh_path,
)
from magitech_harness.shell.render import ParsedExitStatus, parse_exit_status
from magitech_harness.shell.service import LocalShellService
from magitech_harness.shell.types import (
    MAGITECH_ENV_PREFIX,
    CollectedOutput,
    CollectedOutputReader,
    OutputReadSlice,
    ShellExecRequest,
    ShellExecSpec,
    ShellExecution,
    ShellExpiryPolicy,
    ShellObservedStreams,
    ShellProcess,
    ShellProcessRead,
    ShellProcessStatus,
    ShellResult,
    ShellRunResult,
    ShellSandboxInfo,
)

__all__ = [
    "BASH_ENV_OVERRIDES",
    "MAGITECH_ENV_PREFIX",
    "PWSH_ENCODING_PREAMBLE",
    "PWSH_ENV_OVERRIDES",
    "CollectedOutput",
    "CollectedOutputReader",
    "LocalBashExecutor",
    "LocalPwshExecutor",
    "LocalShellService",
    "OutputReadSlice",
    "ParsedExitStatus",
    "PwshLocalExecutor",
    "ShellExecRequest",
    "ShellExecSpec",
    "ShellExecution",
    "ShellExecutor",
    "ShellExpiryPolicy",
    "ShellObservedStreams",
    "ShellProcess",
    "ShellProcessRead",
    "ShellProcessStatus",
    "ShellResult",
    "ShellRunResult",
    "ShellSandboxInfo",
    "assert_serviceable_bash_config",
    "assert_serviceable_pwsh_config",
    "candidate_pwsh_paths",
    "parse_exit_status",
    "resolve_pwsh_path",
]
