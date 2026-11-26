"""Credential service (ctx.credentials): stores, looks up, and removes secrets.

Keeps secret values out of configuration by letting settings refer to key names
such as DEEPSEEK_API_KEY. Stores durable per-plugin credential records.

Reference: ``packages/credentials/credentials``.
"""

from __future__ import annotations

from magitech_harness.credentials.engine import (
    CredentialConfig,
    CredentialInfo,
    CredentialKey,
    CredentialProvider,
    CredentialRecord,
    CredentialRecordInfo,
    CredentialRecordKey,
    CredentialRef,
    CredentialService,
    ResolvedCredential,
)

__all__ = [
    "CredentialConfig",
    "CredentialInfo",
    "CredentialKey",
    "CredentialProvider",
    "CredentialRecord",
    "CredentialRecordInfo",
    "CredentialRecordKey",
    "CredentialRef",
    "CredentialService",
    "ResolvedCredential",
]
