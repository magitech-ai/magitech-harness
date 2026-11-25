"""Session feedback: log-only ``feedback/record`` and the ``/feedback`` command.

Ports ``packages/feedback/command-feedback/src/index.ts``. Recording never
starts model work; it appends one authoritative log-only event to the target
session and acknowledges it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from animas import Context, Service

from magitech_harness.commands.types import (
    CommandDefinition,
    CommandInputDescriptor,
    CommandResult,
)
from magitech_harness.feedback.types import (
    FeedbackRecord,
    SessionFeedbackRecordRequest,
    SessionFeedbackRecordResult,
)

USAGE = "Usage: /feedback <text>"


def record_feedback(session: Any, entry: FeedbackRecord | Mapping[str, Any]) -> None:
    """Append one feedback record; blank text is stored as absent."""
    if isinstance(entry, Mapping):
        entry = FeedbackRecord(text=entry.get("text"), category=entry.get("category"))
    entry.validate()
    text = (entry.text or "").strip()
    session.append(
        "feedback/record",
        {
            **({} if not text else {"text": text}),
            **({} if entry.category is None else {"category": entry.category}),
        },
    )


class SessionFeedbackService(Service):
    """``ctx.sessionFeedback``: command-independent feedback recording."""

    inject: ClassVar[list[str]] = ["sessions"]

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, "sessionFeedback")
        del config

    def record(
        self, request: SessionFeedbackRecordRequest
    ) -> SessionFeedbackRecordResult:
        """Record one remark on a live session."""
        session = self.ctx.sessions.get(request.session_id)
        if session is None:
            return SessionFeedbackRecordResult(
                False,
                {"code": "session-not-found", "sessionId": request.session_id},
            )
        record_feedback(
            session,
            FeedbackRecord(text=request.text, category=request.category),
        )
        return SessionFeedbackRecordResult(True)


def apply(ctx: Context, config: Mapping[str, Any] | None = None) -> None:
    """Mount ``ctx.sessionFeedback`` and register the ``/feedback`` command."""
    del config
    service = ctx.get("sessionFeedback", None, strict=False)
    if service is None:
        service = SessionFeedbackService(ctx)

    def handler(invocation: Any) -> CommandResult:
        raw = invocation.raw_input.strip()
        if not raw:
            return CommandResult.error(f"Feedback text is required. {USAGE}")
        record_feedback(invocation.agent.session, FeedbackRecord(text=raw))
        return CommandResult.success(
            f"Feedback recorded for session {invocation.agent.session.session_id}"
        )

    ctx.commands.register(
        CommandDefinition(
            name="feedback",
            description="Record session feedback",
            handler=handler,
            definition_id="@deepseek-ai/dsh-command-feedback",
            input=CommandInputDescriptor(hint="[text]"),
        )
    )


apply.inject = ["commands", "sessions"]  # type: ignore[attr-defined]

__all__ = ["USAGE", "SessionFeedbackService", "apply", "record_feedback"]
