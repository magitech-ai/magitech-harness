"""Unit tests for public application attribution identity and HTTP headers."""

from __future__ import annotations

from importlib import metadata

from magitech_harness.llm.attribution import (
    APP_IDENTITY,
    AppIdentity,
    attribution_headers,
    user_agent,
)


def test_app_identity_user_agent_and_attribution_headers() -> None:
    pkg_version = metadata.version("magitech-harness")
    assert APP_IDENTITY == {
        "product": "magitech-harness",
        "version": pkg_version,
        "url": "https://github.com/magitech-ai/magitech-harness",
    }
    expected_ua = (
        f"magitech-harness/{pkg_version} "
        "(+https://github.com/magitech-ai/magitech-harness)"
    )
    assert user_agent() == expected_ua
    assert attribution_headers() == {"user-agent": expected_ua}

    fork_identity: AppIdentity = {
        "product": "fork-agent",
        "version": "9.9.9",
        "url": "https://example.com/fork-agent",
    }
    assert (
        user_agent(fork_identity)
        == "fork-agent/9.9.9 (+https://example.com/fork-agent)"
    )
    assert attribution_headers(fork_identity) == {
        "user-agent": "fork-agent/9.9.9 (+https://example.com/fork-agent)"
    }
