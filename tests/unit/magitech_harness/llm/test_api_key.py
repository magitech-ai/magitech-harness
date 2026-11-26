"""Unit tests for API key normalization and refusal diagnostics."""

from __future__ import annotations

import pytest

from magitech_harness.llm.adapter_failure import assert_usable_api_key
from magitech_harness.llm.api_key import normalize_api_key
from magitech_harness.llm.error import INVALID_CREDENTIAL_CODE


def test_normalize_api_key_valid_and_trimmed() -> None:
    assert normalize_api_key("sk-0123456789abcdef") == {
        "ok": True,
        "value": "sk-0123456789abcdef",
    }
    assert normalize_api_key("  sk-abc\t\n") == {"ok": True, "value": "sk-abc"}
    assert normalize_api_key("!~") == {"ok": True, "value": "!~"}


@pytest.mark.parametrize("raw", ["", "   ", "\t"])
def test_normalize_api_key_rejects_empty(raw: str) -> None:
    assert normalize_api_key(raw) == {"ok": False, "reason": "empty"}


@pytest.mark.parametrize(
    "raw",
    [
        "sk-\U0001f600abc",
        "sk-\u03a9abc",
        "sk-abc\u00b7",
        "sk-abc def",
        "sk-abc\x01",
        "sk-café",
    ],
)
def test_normalize_api_key_rejects_illegal_characters(raw: str) -> None:
    assert normalize_api_key(raw) == {"ok": False, "reason": "illegalCharacters"}


def test_assert_usable_api_key_success_and_error_diagnostics() -> None:
    assert assert_usable_api_key("  sk-abc  ", "llm-provider", "API_KEY") == "sk-abc"

    with pytest.raises(Exception) as blank_exc:
        assert_usable_api_key("   ", "llm-provider", "PROVIDER_API_KEY")
    assert "llm-provider: the API key resolved from PROVIDER_API_KEY is blank" in str(
        blank_exc.value
    )
    assert blank_exc.value.code == INVALID_CREDENTIAL_CODE  # type: ignore[attr-defined]

    with pytest.raises(Exception) as illegal_exc:
        assert_usable_api_key("sk-\U0001f600supersecret", "llm-adapter", "ACME_API_KEY")
    msg = str(illegal_exc.value)
    assert illegal_exc.value.code == INVALID_CREDENTIAL_CODE  # type: ignore[attr-defined]
    assert "llm-adapter" in msg
    assert "ACME_API_KEY" in msg
    assert "supersecret" not in msg
