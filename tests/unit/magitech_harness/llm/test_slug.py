"""Unit tests for model slug parsing."""

from magitech_harness.llm.slug import parse_model_slug


def test_parse_model_slug_with_provider_prefix() -> None:
    """Verifies parsing model slug with explicit provider prefix."""
    provider, model = parse_model_slug("openrouter/meta-llama/llama-3-70b")
    assert provider == "openrouter"
    assert model == "meta-llama/llama-3-70b"

    provider2, model2 = parse_model_slug("openai/gpt-4o")
    assert provider2 == "openai"
    assert model2 == "gpt-4o"


def test_parse_model_slug_without_provider() -> None:
    """Verifies parsing model slug without provider prefix."""
    provider, model = parse_model_slug("gpt-4o")
    assert provider == ""
    assert model == "gpt-4o"


def test_parse_model_slug_empty_string() -> None:
    """Verifies parsing empty model slug."""
    provider, model = parse_model_slug("")
    assert provider == ""
    assert model == ""
