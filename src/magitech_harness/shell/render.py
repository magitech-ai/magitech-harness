"""Shared exit-status marker parser for shell tool outputs."""

from __future__ import annotations

import re
from typing import NotRequired, TypedDict

_SIGNAL_RE = re.compile(r"\n\[killed by signal: ([^\]\n]+)\]$")
_EXIT_RE = re.compile(r"\n\[exit code: (\d+)\]$")


class ParsedExitStatus(TypedDict):
    """Split shell tool output separating body text from trailing exit marker."""

    body: str
    exitCode: NotRequired[int]
    signal: NotRequired[str]


def parse_exit_status(text: str) -> ParsedExitStatus:
    """Split trailing ``[killed by signal: X]`` or ``[exit code: N]`` from *text*."""
    sig_match = _SIGNAL_RE.search(text)
    if sig_match is not None:
        return {"body": text[: sig_match.start()], "signal": sig_match.group(1)}
    exit_match = _EXIT_RE.search(text)
    if exit_match is not None:
        return {
            "body": text[: exit_match.start()],
            "exitCode": int(exit_match.group(1)),
        }
    return {"body": text, "exitCode": 0}


__all__ = ["ParsedExitStatus", "parse_exit_status"]
