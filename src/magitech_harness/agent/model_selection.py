"""Agent-owned model selection, durable projection, and scoped switch notices."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypedDict

from anima import Context

from magitech_harness.agent.contracts import (
    AgentOptions,
    ModelSelection,
    ModelSelectionProjection,
    PreStepDecision,
)
from magitech_harness.llm.message import bound_context_summary, create_user_message
from magitech_harness.session.events import EventType, SessionEvent
from magitech_harness.session.projection_registry import (
    ProjectionDefinition,
    WireView,
)
from magitech_harness.session.store import Session


class ModelSelectionProjectionState(TypedDict):
    """Durable model-selection state for one Session."""

    lastUsed: ModelSelection | None
    pending: ModelSelection | None


def _to_selection(
    provider: str, model: str, reasoning_effort: str | None = None
) -> ModelSelection:
    selection: ModelSelection = {"provider": provider, "model": model}
    if reasoning_effort is not None:
        selection["reasoningEffort"] = reasoning_effort
    return selection


def _same_route(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        left["provider"] == right["provider"] and left["model"] == right["model"]
    )


def _same_model_selection(
    left: Mapping[str, Any] | None, right: Mapping[str, Any] | None
) -> bool:
    if left is None or right is None:
        return left is right
    return _same_route(left, right) and left.get("reasoningEffort") == right.get(
        "reasoningEffort"
    )


def _selection_of_header(header: Mapping[str, Any]) -> ModelSelection:
    config = header["config"]
    defaults = header.get("adapterDefaults") or {}
    effort = (
        None
        if defaults.get("reasoningEffort") is True
        else config.get("reasoningEffort")
    )
    return _to_selection(config["provider"], config["model"], effort)


def _validate_selection(value: Any, field: str) -> ModelSelection | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"modelSelection.{field} must be a dict or None")
    provider = value.get("provider")
    model = value.get("model")
    if not isinstance(provider, str) or not provider:
        raise ValueError(f"modelSelection.{field}.provider must be a non-empty string")
    if not isinstance(model, str) or not model:
        raise ValueError(f"modelSelection.{field}.model must be a non-empty string")
    if "reasoningEffort" in value:
        effort = value["reasoningEffort"]
        if not isinstance(effort, str) or not effort:
            raise ValueError(
                f"modelSelection.{field}.reasoningEffort must be a non-empty string"
            )
        return _to_selection(provider, model, effort)
    return _to_selection(provider, model)


def _validate_state(value: Any) -> ModelSelectionProjectionState:
    if not isinstance(value, dict):
        raise TypeError("modelSelection state must be a dict with lastUsed and pending")
    if "lastUsed" not in value or "pending" not in value:
        raise ValueError(
            "modelSelection state must be a dict with lastUsed and pending"
        )
    return {
        "lastUsed": _validate_selection(value["lastUsed"], "lastUsed"),
        "pending": _validate_selection(value["pending"], "pending"),
    }


def _validate_view(value: Any) -> ModelSelectionProjection:
    if not isinstance(value, dict):
        raise TypeError("modelSelection view must be a dict with lastUsed and next")
    if "lastUsed" not in value or "next" not in value:
        raise ValueError("modelSelection view must be a dict with lastUsed and next")
    return {
        "lastUsed": _validate_selection(value["lastUsed"], "lastUsed"),
        "next": _validate_selection(value["next"], "next"),
    }


def _init_model_selection(
    _header: dict[str, Any], _inherited_event_count: int
) -> ModelSelectionProjectionState:
    return {"lastUsed": None, "pending": None}


def _apply_model_selection(
    state: ModelSelectionProjectionState, event: SessionEvent
) -> ModelSelectionProjectionState:
    if event.type == "model/selection":
        if _same_model_selection(state["pending"], event.payload):
            return state
        return {**state, "pending": event.payload}  # type: ignore[typeddict-item]
    if event.type != "request/header":
        return state
    last_used = _selection_of_header(event.payload["header"])
    pending = (
        None if _same_model_selection(state["pending"], last_used) else state["pending"]
    )
    if (
        _same_model_selection(state["lastUsed"], last_used)
        and pending is state["pending"]
    ):
        return state
    return {"lastUsed": last_used, "pending": pending}


model_selection_projection_definition: ProjectionDefinition[
    ModelSelectionProjectionState
] = ProjectionDefinition(
    key="modelSelection",
    state_version=3,
    state_schema=_validate_state,
    init=_init_model_selection,
    apply=_apply_model_selection,
    wire=WireView(
        view_schema=_validate_view,
        view=lambda state: {
            "lastUsed": state["lastUsed"],
            "next": (
                state["pending"] if state["pending"] is not None else state["lastUsed"]
            ),
        },
    ),
)


class AgentModel:
    """Read and record an Agent's durable model choice."""

    def __init__(
        self,
        read_state: Callable[[], ModelSelectionProjectionState],
        session: Session,
        options: AgentOptions,
    ) -> None:
        self._read_state = read_state
        self._session = session
        self._options = options

    def current(self) -> ModelSelection | None:
        """Resolve the choice for the next step without resolving adapter defaults."""
        recorded = self._recorded_selection()
        if recorded is not None:
            return dict(recorded)  # type: ignore[return-value]
        provider = self._options.get("provider")
        model = self._options.get("model")
        if provider and model:
            return _to_selection(provider, model, self._options.get("reasoningEffort"))
        return None

    def select(self, selection: ModelSelection) -> None:
        """Persist a new model selection unless an equal recorded selection is active."""
        if _same_model_selection(self._recorded_selection(), selection):
            return
        self._session.append(
            EventType.MODEL_SELECTION,
            dict(
                _to_selection(
                    selection["provider"],
                    selection["model"],
                    selection.get("reasoningEffort"),
                )
            ),
        )

    def _recorded_selection(self) -> ModelSelection | None:
        state = self._read_state()
        return state["pending"] if state["pending"] is not None else state["lastUsed"]


def _route_label(route: Mapping[str, Any], other: Mapping[str, Any]) -> str:
    if route["provider"] == other["provider"]:
        return str(route["model"])
    return f"{route['provider']}/{route['model']}"


def _model_switch_notice(
    previous: Mapping[str, Any], selected: Mapping[str, Any]
) -> dict[str, Any]:
    from_label = _route_label(previous, selected)
    to_label = _route_label(selected, previous)
    return create_user_message(
        content=[
            {
                "type": "text",
                "text": (
                    "[model changed: assistant turns above this point were generated "
                    f"by {from_label}; the session continues with {to_label}]"
                ),
            }
        ],
        source={
            "kind": "model-selection",
            "form": "notice",
            "summary": bound_context_summary(f"{from_label} \u2192 {to_label}"),
        },
    )


def install_model_selection(agent_ctx: Context) -> Callable[[], Any]:
    """Append a model-switch notice when a step changes provider or model route."""

    async def _on_pre_step(
        payload: dict[str, Any],
        next_fn: Callable[[], Awaitable[PreStepDecision]],
    ) -> PreStepDecision:
        decision = await next_fn()
        signal = payload["signal"]
        if decision["kind"] == "reject" or signal.aborted:
            return decision
        step = payload["step"]
        messages = payload["messages"]
        if len(decision["messages"]) == 0 and (step == 1 or len(messages) > 0):
            return decision
        header = payload["agent"].session.request_header()
        previous = header.get("config") if header is not None else None
        selection = payload["modelSelection"]
        if previous is None or _same_route(selection, previous):
            return decision
        return {
            **decision,
            "messages": [
                *decision["messages"],
                _model_switch_notice(previous, selection),
            ],
        }

    return agent_ctx.on("agent/pre-step", _on_pre_step, prepend=True)
