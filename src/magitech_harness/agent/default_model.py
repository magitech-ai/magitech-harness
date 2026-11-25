"""Default agent model (``ctx.agentDefaultModel``).

Ports dsh's ``@deepseek-ai/dsh-agent-default-model`` at
``packages/core/agent-default-model/src/index.ts``: a small service holding the
transport-independent default provider/model that entry points consult when
creating an agent without an explicit selection.

The shipped base row leaves both fields empty because this stack mounts no model
provider by default (ADR 0002 / ADR 0012); a deployment that mounts a provider
sets them, and creations without ``agentOptions`` then inherit the default.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from animas import Context, Service


class AgentDefaultModel(Service):
    """``ctx.agentDefaultModel``: the default provider/model for new agents."""

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, "agentDefaultModel")
        raw = dict(config or {})
        self._provider = self._text(raw.get("provider"), "provider")
        self._model = self._text(raw.get("model"), "model")

    @staticmethod
    def _text(value: Any, field: str) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise TypeError(f"agent-default-model: {field} must be a string")
        return value

    def get(self) -> dict[str, str] | None:
        """The configured default, or ``None`` when one field is unset."""
        if not self._provider or not self._model:
            return None
        return {"provider": self._provider, "model": self._model}

    def set(self, provider: str, model: str) -> None:
        """Replace the default selection (a settings surface owns persistence)."""
        self._provider = self._text(provider, "provider")
        self._model = self._text(model, "model")

    def apply_to(self, agent_options: Mapping[str, Any] | None) -> dict[str, Any]:
        """Fill only the missing provider/model keys from the default."""
        options = dict(agent_options or {})
        default = self.get()
        if default is None:
            return options
        options.setdefault("provider", default["provider"])
        options.setdefault("model", default["model"])
        return options


__all__ = ["AgentDefaultModel"]
