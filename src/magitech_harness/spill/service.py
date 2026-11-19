"""Spill storage service seam (`ctx.spillStore`).

The spill storage service persists caller-provided text verbatim and returns an
opaque locator, exact byte length, and model-facing retrieval guidance. The
seam owns storage only: no retention policy, no tool-result replacement, no
retrieval/search API. Implementations subclass :class:`SpillStore` and register
as ``ctx.spillStore`` (one implementation per context; loading a second
fails loud).

Abstract spill store service and local backend.
"""

from __future__ import annotations

from typing import Any, TypedDict

from anima import Context, Service

__all__ = [
    "SaveTextSpill",
    "SpillLocator",
    "SpillOwner",
    "SpillRef",
    "SpillSource",
    "SpillStore",
]


class SpillLocator(str):
    """Opaque model-facing handle for one spilled artifact.

    A local backend may use a filesystem path; a remote or database backend
    may use a URI or key. Consumers render it with
    :attr:`SpillRef.retrievalHint`, but do not parse it.
    """


class SpillOwner(TypedDict):
    """Save-time storage namespace for a spilled artifact.

    The session id lets a backend group storage under the producing session,
    but the returned :class:`SpillLocator` is the model-facing handle.
    """

    sessionId: str


class SpillSourceTool(TypedDict):
    """Producer of a spilled artifact: a tool result."""

    kind: str  # "tool"
    toolName: str
    callId: str
    label: str


class SpillSourceSessionReference(TypedDict):
    """Producer of a spilled artifact: a captured session reference."""

    kind: str  # "session-reference"
    sessionId: str
    label: str


# A textual union alias is used so the dict shape is explicit at runtime.
SpillSource = SpillSourceTool | SpillSourceSessionReference


class SaveTextSpill(TypedDict):
    """One request to persist text to a spill artifact."""

    owner: SpillOwner
    source: SpillSource
    suggestedName: str
    content: str


class SpillRef(TypedDict):
    """A saved spill artifact: its locator, byte length, and retrieval guidance."""

    locator: SpillLocator
    bytes: int
    retrievalHint: str


class SpillStore(Service):
    """Abstract spill storage service.

    Subclass, implement :meth:`saveText`, and register as a component — it
    registers as ``ctx.spillStore`` (one implementation per context; loading
    a second fails loud).

    Semantics every implementation must honor:

    - :meth:`saveText` persists the full ``content`` verbatim and returns an
      opaque locator, exact byte length, and model-facing retrieval guidance.
    - Storage is scoped by the request's ``owner`` session; the backend chooses
      a private location and a collision-free name derived from — never equal
      to — the caller's ``suggestedName``.
    - :meth:`saveText` rejects on a real storage failure (permissions,
      ENOSPC, backend unavailable); the caller decides how to degrade.
    """

    def __init__(self, ctx: Context) -> None:
        if ctx.get("spillStore", None, strict=False) is not None:
            raise RuntimeError("spillStore service already registered")
        super().__init__(ctx, "spillStore")

    async def saveText(self, input: SaveTextSpill) -> SpillRef:
        """Persist ``input.content`` to a session-scoped spill artifact.

        Args:
            input: the owner, caller-supplied source fields, suggested name,
                and full text to save.

        Returns:
            the saved artifact's :class:`SpillRef`; rejects on a storage
            failure.
        """
        raise NotImplementedError


# Backwards-compatible plain dict alias for callers that build requests
# without TypedDict syntax.
SaveTextSpillDict = dict[str, Any]
