"""Session feedback vocabulary (dsh ``@deepseek-ai/dsh-command-feedback``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

FeedbackCategory = Literal[
    "task-result",
    "instruction-following",
    "product-interaction",
    "service-stability",
    "resource-cost",
    "security-privacy-permission",
    "other",
]

FEEDBACK_CATEGORIES: tuple[str, ...] = (
    "task-result",
    "instruction-following",
    "product-interaction",
    "service-stability",
    "resource-cost",
    "security-privacy-permission",
    "other",
)


@dataclass(frozen=True)
class FeedbackRecord:
    """A human-authored remark about one session."""

    text: str | None = None
    category: str | None = None

    def validate(self) -> None:
        if self.category is not None and self.category not in FEEDBACK_CATEGORIES:
            raise ValueError(f"unknown feedback category {self.category!r}")


@dataclass(frozen=True)
class SessionFeedbackRecordRequest:
    """Request to record feedback against a live session."""

    session_id: str
    text: str | None = None
    category: str | None = None


@dataclass(frozen=True)
class SessionFeedbackRecordResult:
    """Postcondition of one recorded feedback entry."""

    recorded: bool
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **({"recorded": True} if self.recorded else {"recorded": False}),
            **({} if self.error is None else {"error": self.error}),
        }


__all__ = [
    "FEEDBACK_CATEGORIES",
    "FeedbackCategory",
    "FeedbackRecord",
    "SessionFeedbackRecordRequest",
    "SessionFeedbackRecordResult",
]
