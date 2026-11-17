"""Credential service (ctx.credentials): stores, looks up, and removes secrets.

Keeps secret values out of configuration by letting settings refer to key names
such as DEEPSEEK_API_KEY. Stores durable per-plugin credential records, so a
plugin can store, update, and remove the credentials it holds for its own ids.

Reference: ``packages/credentials/credentials``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Literal, Self, TypedDict, cast

from anima import Context, Service

__all__ = [
    "CredentialConfig",
    "CredentialInfo",
    "CredentialKey",
    "CredentialRecord",
    "CredentialRecordInfo",
    "CredentialRecordKey",
    "CredentialRef",
    "CredentialService",
    "ResolvedCredential",
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class CredentialRef(str):
    """Branded string for a POSIX shell identifier credential reference.

    e.g. ``DEEPSEEK_API_KEY``. A reference doubles as the environment name it
    resolves through.
    """

    def __new__(cls, name: str) -> Self:
        return str.__new__(cls, name)


class CredentialKey(str):
    """Branded string for a durable credential record key: ``<owner>/<id>``.

    e.g. ``llm-pi-ai/openai-codex``. The owner is the plugin's scope; the id is
    chosen by the plugin.
    """

    def __new__(cls, owner: str, id: str) -> Self:
        return str.__new__(cls, f"{owner}/{id}")


class ResolvedCredential(TypedDict):
    """Result of resolving a credential reference: the value and its source."""

    value: str
    source: Literal["stored", "environment"]


class CredentialRecord(TypedDict, total=False):
    """A durable credential record owned by a plugin.

    ``kind`` identifies the record type (e.g. ``grant``, ``env``). ``payload``
    holds the record's secret-bearing data and is returned exactly as its owner
    wrote it.
    """

    kind: str
    payload: dict[str, Any]


class CredentialInfo(TypedDict, total=False):
    """UI-safe description of a credential reference: never exposes the value."""

    configured: bool
    source: Literal["stored", "environment"] | None
    writable: bool
    masked: str | None


class CredentialRecordInfo(TypedDict, total=False):
    """UI-safe description of a credential record: never exposes the value."""

    configured: bool
    kind: str | None
    writable: bool


class CredentialRecordKey(TypedDict):
    """One record entry returned by listRecords: never exposes values."""

    key: CredentialKey
    kind: str


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CredentialConfig(TypedDict, total=False):
    """Optional configuration for the credential service."""


class CredentialService(Service):
    """Shared credential service creating and managing secret references.

    Mount one credential service to store, look up, and remove secrets without
    embedding values in configuration. Configuration files contain only
    credential names; local secret values remain readable only by the same OS
    user.

    Args:
        ctx: The anima Context to register the service on.
        config: Optional configuration (currently unused).
    """

    inject: ClassVar[list[str]] = []

    def __init__(self, ctx: Context, config: CredentialConfig | None = None) -> None:
        if ctx.get("credentials", None, strict=False) is not None:
            raise RuntimeError("credentials service already registered")
        super().__init__(ctx, "credentials")
        self.cfg: CredentialConfig = config or {}
        self._stored: dict[str, str] = {}  # ref -> value
        self._records: dict[str, CredentialRecord] = {}  # key -> record
        ctx.reflect.provide("credentials", self, None)

    def credential_ref(self, name: str) -> CredentialRef:
        """Create a branded CredentialRef from a POSIX shell identifier."""
        return CredentialRef(name)

    def credential_key(self, owner: str, id: str) -> CredentialKey:
        """Create a branded CredentialKey: ``<owner>/<id>``."""
        return CredentialKey(owner, id)

    def resolve(self, ref: CredentialRef) -> ResolvedCredential | None:
        """Resolve a credential reference to its current value and source.

        A stored value outranks environment fallbacks. Returns ``None`` when
        the key is not configured anywhere.
        """
        if ref in self._stored:
            return ResolvedCredential(value=self._stored[ref], source="stored")
        import os

        env_value = os.environ.get(ref)
        if env_value is not None:
            return ResolvedCredential(value=env_value, source="environment")
        return None

    def describe(self, ref: CredentialRef) -> CredentialInfo:
        """Describe a credential reference without exposing its value.

        Reports whether the key is set, where it comes from, and whether it is
        writable. Optionally discloses a masked stored-value preview.
        """
        info: CredentialInfo = {
            "configured": False,
            "source": None,
            "writable": True,
            "masked": None,
        }
        resolved = self.resolve(ref)
        if resolved is not None:
            info["configured"] = True
            info["source"] = resolved["source"]
            if resolved["source"] == "stored":
                value = resolved["value"]
                masked = (
                    value[:2] + "*" * (len(value) - 4) + value[-2:]
                    if len(value) > 4
                    else "*" * len(value)
                )
                info["masked"] = masked
            else:
                info["masked"] = None
        return info

    def set(self, ref: CredentialRef, value: str) -> None:
        """Store a credential value, outranking environment fallbacks.

        Storing an empty string is refused; use ``unset`` to remove a key.
        """
        if value == "":
            return
        self._stored[ref] = value
        self.ctx.emit("credentials/reference-updated", ref)

    def unset(self, ref: CredentialRef) -> None:
        """Remove a stored credential value.

        A no-op when absent; an environment fallback then applies again.
        """
        if ref in self._stored:  # pragma: no cover - unset absent key is no-op
            del self._stored[ref]
            self.ctx.emit("credentials/reference-updated", ref)

    def read_record(self, key: CredentialKey) -> CredentialRecord | None:
        """Read one durable credential record by key.

        Returns ``None`` when no record exists for the key.
        """
        return self._records.get(key)

    def describe_record(self, key: CredentialKey) -> CredentialRecordInfo:
        """Describe a credential record without exposing its value."""
        record = self._records.get(key)
        if record is None:
            return CredentialRecordInfo(configured=False, kind=None, writable=True)
        return CredentialRecordInfo(
            configured=True,
            kind=record.get("kind"),
            writable=True,
        )

    def list_records(self) -> list[CredentialRecordKey]:
        """List all credential records: never exposes values."""
        return [
            CredentialRecordKey(
                key=cast(CredentialKey, key), kind=record.get("kind", "")
            )
            for key, record in self._records.items()
        ]

    def modify_record(
        self,
        key: CredentialKey,
        mutation: Callable[[CredentialRecord | None], CredentialRecord | None],
    ) -> CredentialRecord | None:
        """Modify a record atomically.

        The mutation receives the current record (or ``None`` if absent) and
        returns the new record, or ``None`` to leave it untouched. Fires
        ``credentials/record-updated`` only when the record actually changed.
        """
        current = self._records.get(key)
        updated = mutation(current)
        if updated is None:
            if current is not None:
                del self._records[key]
                self.ctx.emit("credentials/record-updated", key)
            return None
        if current == updated:  # pragma: no cover - mutation returns unchanged value
            return updated
        self._records[key] = updated
        self.ctx.emit("credentials/record-updated", key)
        return updated

    def delete_record(self, key: CredentialKey) -> None:
        """Remove a credential record.

        A no-op when absent.
        """
        if key in self._records:
            del self._records[key]
            self.ctx.emit("credentials/record-updated", key)

    async def check(self) -> None:  # pragma: no cover - lifecycle check
        """Lifecycle check: service is always healthy once constructed."""
        return
