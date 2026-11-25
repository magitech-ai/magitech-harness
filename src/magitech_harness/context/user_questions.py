"""Human-question service and the ``ask_user_question`` tool.

Ports dsh's ``@deepseek-ai/dsh-user-questions`` and
``@deepseek-ai/dsh-tool-ask-user`` at
``packages/interaction/{user-questions,tool-ask-user}/src``.

The service is the seam a UI answerer registers on. A tool call appends a
``user/question`` session event (so the question is reconstructable from the
log), then awaits a registered answerer. A timed variant returns ``pending:
true`` after its window closes instead of blocking forever; the answer may
still arrive as a later user message.
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from animas import Context, Service

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.llm.error import HarnessError
from magitech_harness.tools.schema import define_tool

DEFAULT_TIMEOUT_SECONDS = 120


class UserQuestionError(HarnessError):
    """Stable error taxonomy for user-question failures."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message, code)


@dataclass(frozen=True)
class Answer:
    """One answer to one question."""

    id: str
    selected: list[str] = field(default_factory=list)
    custom: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"id": self.id, "selected": list(self.selected)}
        if self.custom is not None:
            value["custom"] = self.custom
        return value


@dataclass
class Pending:
    """One awaiting question batch."""

    call_id: str
    questions: list[dict[str, Any]]
    future: asyncio.Future[list[Answer]]
    session: Any = None


Answerer = Callable[
    [Pending, AbortSignal | None], Awaitable[list[Answer]] | list[Answer]
]


def _validate_questions(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise UserQuestionError(
            "questions must be a non-empty array", "ASK_INVALID_QUESTIONS"
        )
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise UserQuestionError(
                f"questions[{index}] must be an object", "ASK_INVALID_QUESTIONS"
            )
        question_id = item.get("id")
        text = item.get("question")
        if not isinstance(question_id, str) or not question_id.strip():
            raise UserQuestionError(
                f"questions[{index}].id must be a non-empty string",
                "ASK_INVALID_QUESTIONS",
            )
        if question_id in seen:
            raise UserQuestionError(
                f"questions[{index}].id duplicates {question_id!r}",
                "ASK_INVALID_QUESTIONS",
            )
        seen.add(question_id)
        if not isinstance(text, str) or not text.strip():
            raise UserQuestionError(
                f"questions[{index}].question must be a non-empty string",
                "ASK_INVALID_QUESTIONS",
            )
        normalized: dict[str, Any] = {
            "id": question_id,
            "question": text,
            "multiSelect": bool(item.get("multi_select", False)),
        }
        if "header" in item and item["header"] is not None:
            if not isinstance(item["header"], str):
                raise UserQuestionError(
                    f"questions[{index}].header must be a string",
                    "ASK_INVALID_QUESTIONS",
                )
            normalized["header"] = item["header"]
        options = item.get("options")
        if options is not None:
            if not isinstance(options, list) or not options:
                raise UserQuestionError(
                    f"questions[{index}].options must be a non-empty array when given",
                    "ASK_INVALID_QUESTIONS",
                )
            normalized_options: list[dict[str, str]] = []
            for option_index, option in enumerate(options):
                if (
                    not isinstance(option, Mapping)
                    or not isinstance(option.get("label"), str)
                    or not option["label"]
                ):
                    raise UserQuestionError(
                        f"questions[{index}].options[{option_index}].label must be a "
                        "non-empty string",
                        "ASK_INVALID_QUESTIONS",
                    )
                value = {"label": option["label"]}
                if "description" in option and option["description"] is not None:
                    if not isinstance(option["description"], str):
                        raise UserQuestionError(
                            f"questions[{index}].options[{option_index}].description "
                            "must be a string",
                            "ASK_INVALID_QUESTIONS",
                        )
                    value["description"] = option["description"]
                normalized_options.append(value)
            normalized["options"] = normalized_options
        questions.append(normalized)
    return questions


class UserQuestionService(Service):
    """``ctx.userQuestions``: validation plus the registered answerer seam."""

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, "userQuestions")
        self._answerers: list[Answerer] = []

    def register_answerer(self, answerer: Answerer) -> Callable[[], None]:
        """Register a UI answerer; returns its disposer."""
        self._answerers.append(answerer)

        def dispose() -> None:
            if answerer in self._answerers:
                self._answerers.remove(answerer)

        return dispose

    async def ask(
        self,
        questions: list[dict[str, Any]],
        *,
        agent: Any = None,
        signal: AbortSignal | None = None,
    ) -> list[Answer]:
        """Append the question, then await the first registered answerer."""
        normalized = _validate_questions(questions)
        call_id = f"ask-{uuid.uuid4().hex[:12]}"
        pending = Pending(
            call_id=call_id,
            questions=normalized,
            future=asyncio.get_running_loop().create_future(),
        )
        session = getattr(agent, "session", None)
        pending.session = session
        if session is not None:
            session.append(
                "user/question",
                {"callId": call_id, "questions": copy.deepcopy(normalized)},
            )
        if not self._answerers:
            raise UserQuestionError(
                "no user-question answerer is registered", "ASK_UNAVAILABLE"
            )
        try:
            result = self._answerers[-1](pending, signal)
            if hasattr(result, "__await__"):
                result = await result
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise UserQuestionError(str(error), "ASK_FAILED") from error
        return self._validate_answers(result, normalized)

    @staticmethod
    def _validate_answers(result: Any, questions: list[dict[str, Any]]) -> list[Answer]:
        if not isinstance(result, (list, tuple)):
            raise UserQuestionError(
                "answerer must return a list of answers", "ASK_INVALID_ANSWER"
            )
        expected = {question["id"] for question in questions}
        answers: list[Answer] = []
        seen: set[str] = set()
        for item in result:
            if isinstance(item, Answer):
                item = item.to_dict()
            if not isinstance(item, Mapping) or item.get("id") not in expected:
                raise UserQuestionError(
                    "answer id does not match a question", "ASK_INVALID_ANSWER"
                )
            answer_id = item["id"]
            if answer_id in seen:
                raise UserQuestionError(
                    f"duplicate answer for {answer_id!r}", "ASK_INVALID_ANSWER"
                )
            seen.add(answer_id)
            selected = item.get("selected", [])
            if not isinstance(selected, list) or not all(
                isinstance(value, str) for value in selected
            ):
                raise UserQuestionError(
                    "answer.selected must be an array of strings", "ASK_INVALID_ANSWER"
                )
            custom = item.get("custom")
            if custom is not None and not isinstance(custom, str):
                raise UserQuestionError(
                    "answer.custom must be a string", "ASK_INVALID_ANSWER"
                )
            answers.append(Answer(id=answer_id, selected=selected, custom=custom))
        return answers


_QUESTION_SCHEMA = {
    "type": "array",
    "required": True,
    "description": "Questions to ask the user before continuing.",
    "items": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "id": {
                "type": "string",
                "required": True,
                "description": "Stable id for this question; echoed in the answer.",
            },
            "question": {
                "type": "string",
                "required": True,
                "description": "The specific question to ask the user.",
            },
            "header": {
                "type": "string",
                "description": "Optional short heading, such as 'Confirm'.",
            },
            "options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "label": {
                            "type": "string",
                            "required": True,
                            "description": "Short user-facing option label.",
                        },
                        "description": {"type": "string"},
                    },
                },
            },
            "multi_select": {"type": "boolean"},
        },
    },
}

_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answers": {
            "type": "array",
            "required": True,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "required": True},
                    "selected": {
                        "type": "array",
                        "required": True,
                        "items": {"type": "string"},
                    },
                    "custom": {"type": "string"},
                },
            },
        },
    },
}


def apply(ctx: Context, config: Mapping[str, Any] | None = None) -> None:
    """Mount ``ctx.userQuestions`` and the blocking ``ask_user_question`` tool."""
    del config
    service = ctx.get("userQuestions", None, strict=False)
    if service is None:
        service = UserQuestionService(ctx)

    async def _execute(args: dict[str, Any], exec_view: Any) -> dict[str, Any]:
        answers = await service.ask(
            args.get("questions", []),
            agent=exec_view.get("agent"),
            signal=exec_view.get("signal"),
        )
        return {"answers": [answer.to_dict() for answer in answers]}

    ctx.tools.register(
        define_tool(
            name="ask_user_question",
            description=(
                "Ask the user a concise question when you need confirmation, a "
                "choice, or missing information before proceeding. A skipped "
                "answer is reported as an empty selection."
            ),
            parameters={"questions": _QUESTION_SCHEMA},
            output={
                "schema": _OUTPUT_SCHEMA,
                "render": lambda _args, value: [
                    {
                        "type": "text",
                        "text": __import__("json").dumps(value),
                    }
                ],
            },
            execute=_execute,
        )
    )


apply.inject = ["tools"]  # type: ignore[attr-defined]

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "Answer",
    "Pending",
    "UserQuestionError",
    "UserQuestionService",
    "apply",
]
