"""Unit tests for credential service."""

from __future__ import annotations

import os
from unittest.mock import patch

from magitech_harness.credentials.engine import (
    CredentialConfig,
    CredentialInfo,
    CredentialKey,
    CredentialRecord,
    CredentialRecordInfo,
    CredentialRef,
    CredentialService,
    ResolvedCredential,
)


def test_credential_ref() -> None:
    """CredentialRef is a branded string."""
    ref = CredentialRef("DEEPSEEK_API_KEY")
    assert ref == "DEEPSEEK_API_KEY"
    assert isinstance(ref, str)


def test_credential_key() -> None:
    """CredentialKey is a branded string: owner/id."""
    key = CredentialKey("llm-pi-ai", "openai-codex")
    assert key == "llm-pi-ai/openai-codex"
    assert isinstance(key, str)


def test_service_creation() -> None:
    """CredentialService registers itself as ctx.credentials."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    assert ctx.get("credentials", None, strict=False) is service


def test_service_duplicate_raises() -> None:
    """Creating CredentialService twice raises RuntimeError."""
    from animas import Context

    ctx = Context()
    CredentialService(ctx)
    try:
        CredentialService(ctx)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


def test_resolve_stored_value() -> None:
    """resolve() returns stored value with source 'stored'."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    ref = service.credential_ref("TEST_KEY")
    service.set(ref, "secret-value")
    result = service.resolve(ref)
    assert result == ResolvedCredential(value="secret-value", source="stored")


def test_resolve_environment_value() -> None:
    """resolve() returns environment value when not stored."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    ref = service.credential_ref("ENV_ONLY_KEY")
    with patch.dict(os.environ, {"ENV_ONLY_KEY": "env-value"}, clear=False):
        result = service.resolve(ref)
        assert result == ResolvedCredential(value="env-value", source="environment")


def test_resolve_not_found() -> None:
    """resolve() returns None when key is not configured."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    ref = service.credential_ref("MISSING_KEY")
    result = service.resolve(ref)
    assert result is None


def test_stored_outranks_environment() -> None:
    """Stored value outranks environment fallback."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    ref = service.credential_ref("OVERRIDE_KEY")
    with patch.dict(os.environ, {"OVERRIDE_KEY": "env-value"}, clear=False):
        service.set(ref, "stored-value")
        result = service.resolve(ref)
        assert result == ResolvedCredential(value="stored-value", source="stored")


def test_describe_stored() -> None:
    """describe() reports stored key as configured with masked value."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    ref = service.credential_ref("DESCRIBE_KEY")
    service.set(ref, "secret-value-12345")
    info = service.describe(ref)
    assert info == CredentialInfo(
        configured=True, source="stored", writable=True, masked="se**************45"
    )


def test_describe_environment() -> None:
    """describe() reports environment key as configured."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    ref = service.credential_ref("ENV_DESCRIBE_KEY")
    with patch.dict(os.environ, {"ENV_DESCRIBE_KEY": "env-value"}, clear=False):
        info = service.describe(ref)
        assert info == CredentialInfo(
            configured=True, source="environment", writable=True, masked=None
        )


def test_describe_not_configured() -> None:
    """describe() reports unconfigured key."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    ref = service.credential_ref("UNCONFIGURED_KEY")
    info = service.describe(ref)
    assert info == CredentialInfo(
        configured=False, source=None, writable=True, masked=None
    )


def test_set_and_unset() -> None:
    """set() stores a value, unset() removes it."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    ref = service.credential_ref("SET_UNSET_KEY")
    service.set(ref, "temp-value")
    assert service.resolve(ref) is not None
    service.unset(ref)
    assert service.resolve(ref) is None


def test_set_empty_value_refused() -> None:
    """set() with empty string is refused."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    ref = service.credential_ref("EMPTY_KEY")
    service.set(ref, "")
    assert service.resolve(ref) is None


def test_read_record() -> None:
    """read_record() returns stored record by key."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    key = service.credential_key("llm-pi-ai", "openai-codex")
    service.modify_record(
        key, lambda _: CredentialRecord(kind="grant", payload={"token": "abc"})
    )
    record = service.read_record(key)
    assert record == CredentialRecord(kind="grant", payload={"token": "abc"})


def test_read_record_not_found() -> None:
    """read_record() returns None when record does not exist."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    key = service.credential_key("llm-pi-ai", "missing-key")
    record = service.read_record(key)
    assert record is None


def test_describe_record() -> None:
    """describe_record() reports record info without exposing value."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    key = service.credential_key("llm-pi-ai", "openai-codex")
    info = service.describe_record(key)
    assert info == CredentialRecordInfo(configured=False, kind=None, writable=True)

    service.modify_record(
        key, lambda _: CredentialRecord(kind="grant", payload={"token": "abc"})
    )
    info = service.describe_record(key)
    assert info == CredentialRecordInfo(configured=True, kind="grant", writable=True)


def test_list_records() -> None:
    """list_records() returns all records without exposing values."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    key1 = service.credential_key("llm-pi-ai", "openai-codex")
    key2 = service.credential_key("llm-pi-ai", "anthropic-codex")
    service.modify_record(
        key1, lambda _: CredentialRecord(kind="grant", payload={"token": "abc"})
    )
    service.modify_record(
        key2, lambda _: CredentialRecord(kind="env", payload={"key": "xyz"})
    )
    records = service.list_records()
    record_keys = {r["key"] for r in records}
    assert key1 in record_keys
    assert key2 in record_keys
    assert len(records) == 2


def test_modify_record() -> None:
    """modify_record() updates a record atomically."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    key = service.credential_key("llm-pi-ai", "openai-codex")
    result = service.modify_record(
        key,
        lambda current: (
            CredentialRecord(kind="grant", payload={"token": "new-token"})
            if current is None or current.get("kind") != "grant"
            else None
        ),
    )
    assert result == CredentialRecord(kind="grant", payload={"token": "new-token"})


def test_modify_record_no_change() -> None:
    """modify_record() returns None when mutation returns None."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    key = service.credential_key("llm-pi-ai", "openai-codex")
    result = service.modify_record(key, lambda _: None)
    assert result is None


def test_modify_record_remove_existing() -> None:
    """modify_record() removes existing record when mutation returns None."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    key = service.credential_key("llm-pi-ai", "openai-codex")
    service.modify_record(
        key, lambda _: CredentialRecord(kind="grant", payload={"token": "abc"})
    )
    assert service.read_record(key) is not None
    result = service.modify_record(key, lambda _: None)
    assert result is None
    assert service.read_record(key) is None


def test_delete_record() -> None:
    """delete_record() removes a record."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    key = service.credential_key("llm-pi-ai", "openai-codex")
    service.modify_record(
        key, lambda _: CredentialRecord(kind="grant", payload={"token": "abc"})
    )
    assert service.read_record(key) is not None
    service.delete_record(key)
    assert service.read_record(key) is None


def test_delete_record_not_found() -> None:
    """delete_record() is a no-op when record does not exist."""
    from animas import Context

    ctx = Context()
    service = CredentialService(ctx)
    key = service.credential_key("llm-pi-ai", "missing-key")
    service.delete_record(key)  # Should not raise


def test_credential_config_typed_dict() -> None:
    """CredentialConfig accepts optional fields."""
    config: CredentialConfig = {}
    assert config == {}
