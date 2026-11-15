"""Unit tests for magitech_harness.shell.render."""

from __future__ import annotations

from magitech_harness.shell.render import parse_exit_status


def test_parse_exit_status() -> None:
    assert parse_exit_status("hello\nworld") == {
        "body": "hello\nworld",
        "exitCode": 0,
    }
    assert parse_exit_status("out\n[exit code: 2]") == {
        "body": "out",
        "exitCode": 2,
    }
    assert parse_exit_status("out\n[killed by signal: SIGTERM]") == {
        "body": "out",
        "signal": "SIGTERM",
    }
    # Mid-body or non-matching marker stays in body
    assert parse_exit_status("first\n[exit code: 2]\nmore") == {
        "body": "first\n[exit code: 2]\nmore",
        "exitCode": 0,
    }
