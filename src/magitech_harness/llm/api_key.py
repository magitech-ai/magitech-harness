"""Validation and trimming for supplied provider API keys."""

from __future__ import annotations

import re
from typing import Literal, TypedDict

_LEGAL_API_KEY = re.compile(r"^[\x21-\x7E]+$")

ApiKeyRejection = Literal["empty", "illegalCharacters"]


class ApiKeyValid(TypedDict):
    """Accepted trimmed API key."""

    ok: Literal[True]
    value: str


class ApiKeyInvalid(TypedDict):
    """Rejected API key with machine-readable reason."""

    ok: Literal[False]
    reason: ApiKeyRejection


ApiKeyCheck = ApiKeyValid | ApiKeyInvalid


def normalize_api_key(raw: str) -> ApiKeyCheck:
    """Trim surrounding whitespace and validate printable ASCII HTTP-header characters."""
    value = raw.strip()
    if not value:
        return {"ok": False, "reason": "empty"}
    if not _LEGAL_API_KEY.fullmatch(value):
        return {"ok": False, "reason": "illegalCharacters"}
    return {"ok": True, "value": value}


normalizeApiKey = normalize_api_key
