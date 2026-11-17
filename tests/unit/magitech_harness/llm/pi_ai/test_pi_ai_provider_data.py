"""Unit tests for the pi-ai provider catalog metadata."""

from __future__ import annotations

from magitech_harness.llm.pi_ai.provider_data import (
    LANES,
    PROVIDER_DATA,
    SUPPORTED_APIS,
)


def test_provider_data_ships_every_catalog_entry() -> None:
    """Verifies the catalog ships all forty-two provider entries."""
    assert len(PROVIDER_DATA) == 42
    assert all("env" in data for data in PROVIDER_DATA.values())


def test_provider_data_carries_literal_ids_env_and_urls() -> None:
    """Verifies the catalog keeps literal provider ids, env names, and URLs."""
    assert PROVIDER_DATA["deepseek"] == {
        "env": ["DEEPSEEK_API_KEY"],
        "defaultBaseURL": "https://api.deepseek.com",
    }
    assert PROVIDER_DATA["openrouter"]["defaultBaseURL"] == (
        "https://openrouter.ai/api/v1"
    )
    assert PROVIDER_DATA["openai"]["defaultBaseURL"] == "https://api.openai.com/v1"
    assert PROVIDER_DATA["anthropic"]["defaultBaseURL"] == "https://api.anthropic.com"
    assert PROVIDER_DATA["google"]["defaultBaseURL"] == (
        "https://generativelanguage.googleapis.com/v1beta"
    )
    assert PROVIDER_DATA["groq"]["defaultBaseURL"] == "https://api.groq.com/openai/v1"
    assert PROVIDER_DATA["xai"]["defaultBaseURL"] == "https://api.x.ai/v1"
    assert PROVIDER_DATA["opencode"]["env"] == ["OPENCODE_API_KEY"]
    assert "defaultBaseURL" not in PROVIDER_DATA["opencode"]


def test_provider_data_declares_base_url_environments() -> None:
    """Verifies entries that read their endpoint from the environment say so."""
    assert PROVIDER_DATA["azure"]["baseURLEnv"] == [
        "AZURE_OPENAI_BASE_URL",
        "AZURE_OPENAI_RESOURCE_NAME",
    ]
    assert PROVIDER_DATA["amazon-bedrock"]["baseURLEnv"] == [
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ]


def test_default_lanes_and_supported_protocols() -> None:
    """Verifies the default lanes and the four served wire protocols."""
    assert LANES == (
        "openrouter",
        "opencode",
        "anthropic",
        "openai",
        "google",
        "groq",
        "xai",
    )
    assert (
        frozenset(
            {
                "openai-completions",
                "openai-responses",
                "anthropic-messages",
                "google-generative-ai",
            }
        )
        == SUPPORTED_APIS
    )
