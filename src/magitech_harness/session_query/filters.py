"""Provider-independent session and event filters.  # pragma: no cover
  # pragma: no cover
Filter arrays are ANDed; values within one list clause are ORED. Ranges are  # pragma: no cover
inclusive. Empty list values match nothing. The event ``text`` clause is a  # pragma: no cover
literal, case-insensitive, whitespace-flexible scan over extracted semantic  # pragma: no cover
text.  # pragma: no cover
Reference: ``packages/session-query/session-query/src/filters.ts``.  # pragma: no cover
"""  # pragma: no cover

# pragma: no cover
from __future__ import annotations  # pragma: no cover

# pragma: no cover
import re  # pragma: no cover
from collections.abc import Callable  # pragma: no cover
from typing import Any, Literal, TypedDict, cast  # pragma: no cover

# pragma: no cover
from magitech_harness.session.events import SessionEvent  # pragma: no cover
from magitech_harness.session_query.errors import invalid_filter  # pragma: no cover
from magitech_harness.session_query.extraction import (
    extract_semantic_text,  # pragma: no cover
)
from magitech_harness.session_query.types import (  # pragma: no cover
    SessionRecord,  # pragma: no cover
    SessionResultRange,  # pragma: no cover
)  # pragma: no cover

# pragma: no cover
__all__ = [  # pragma: no cover
    "SessionEventResultFilter",  # pragma: no cover
    "SessionResultFilter",  # pragma: no cover
    "compile_event_filter",  # pragma: no cover
    "compile_session_filter",  # pragma: no cover
]  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionResultFilter(TypedDict, total=False):  # pragma: no cover
    """One logical-session predicate. ANDed; values within a clause are ORED."""  # pragma: no cover

    # pragma: no cover
    kind: LiteralFilterKind  # pragma: no cover
    values: list[Any]  # pragma: no cover


# pragma: no cover
# pragma: no cover
LiteralFilterKind = Literal[  # pragma: no cover
    "id",  # pragma: no cover
    "cwd",  # pragma: no cover
    "created-at",  # pragma: no cover
    "parent",  # pragma: no cover
    "availability",  # pragma: no cover
    "seq",  # pragma: no cover
    "time",  # pragma: no cover
    "type",  # pragma: no cover
    "surface",  # pragma: no cover
    "text",  # pragma: no cover
]  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionEventResultFilter(TypedDict, total=False):  # pragma: no cover
    """One event predicate. ANDed; list-valued clauses are ORED."""  # pragma: no cover

    # pragma: no cover
    kind: LiteralFilterKind  # pragma: no cover
    values: list[Any] | None  # pragma: no cover
    gte: Any | None  # pragma: no cover
    lte: Any | None  # pragma: no cover
    text: str | None  # pragma: no cover


# pragma: no cover
# pragma: no cover
def _validate_range(kind: str, rng: SessionResultRange) -> None:  # pragma: no cover
    """Reject malformed ranges."""  # pragma: no cover
    if not isinstance(rng, dict):  # pragma: no cover
        raise invalid_filter(f"{kind} range must be an object")  # pragma: no cover
    gte = rng.get("gte")  # pragma: no cover
    lte = rng.get("lte")  # pragma: no cover
    if gte is not None and lte is not None:  # pragma: no cover
        try:  # pragma: no cover
            if float(gte) > float(lte):  # pragma: no cover - type: ignore[arg-type]
                raise invalid_filter(
                    f"{kind} range gte ({gte}) exceeds lte ({lte})"
                )  # pragma: no cover
        except (TypeError, ValueError):  # pragma: no cover
            raise invalid_filter(
                f"{kind} range bounds are not comparable"
            )  # pragma: no cover


# pragma: no cover
# pragma: no cover
def _range_matches(value: Any, rng: SessionResultRange) -> bool:  # pragma: no cover
    """Check one value against an inclusive range."""  # pragma: no cover
    gte = rng.get("gte")  # pragma: no cover
    lte = rng.get("lte")  # pragma: no cover
    if gte is not None:  # pragma: no cover
        try:  # pragma: no cover
            if float(value) < float(gte):  # pragma: no cover - type: ignore[arg-type]
                return False  # pragma: no cover
        except (TypeError, ValueError):  # pragma: no cover
            return False  # pragma: no cover
    if lte is not None:  # pragma: no cover
        try:  # pragma: no cover
            if float(value) > float(lte):  # pragma: no cover - type: ignore[arg-type]
                return False  # pragma: no cover
        except (TypeError, ValueError):  # pragma: no cover
            return False  # pragma: no cover
    return True  # pragma: no cover


# pragma: no cover
# pragma: no cover
def _list_matches(value: Any, values: list[Any]) -> bool:  # pragma: no cover
    """Check one value against an ORed list clause."""  # pragma: no cover
    if not values:  # pragma: no cover
        return False  # pragma: no cover
    if value in values:  # pragma: no cover
        return True  # pragma: no cover
    if value is None:  # pragma: no cover
        return any(v is None for v in values)  # pragma: no cover
    return any(v == value for v in values)  # pragma: no cover


# pragma: no cover
# pragma: no cover
def _event_document(  # pragma: no cover
    event: SessionEvent,  # pragma: no cover
    surface: Literal["current", "shadowed", "log-only"],  # pragma: no cover
    extracted_text: str | None = None,  # pragma: no cover
) -> dict[str, Any]:  # pragma: no cover
    """Build one filterable event document."""  # pragma: no cover
    return {  # pragma: no cover
        "sessionId": event.session_id,  # pragma: no cover
        "seq": event.seq,  # pragma: no cover
        "type": event.type,  # pragma: no cover
        "time": int(event.timestamp * 1000),  # pragma: no cover
        "surface": surface,  # pragma: no cover
        "text": extracted_text  # pragma: no cover
        if extracted_text is not None  # pragma: no cover
        else extract_semantic_text(event),  # pragma: no cover
    }  # pragma: no cover


# pragma: no cover
# pragma: no cover
def compile_session_filter(  # pragma: no cover
    filters: list[SessionResultFilter],  # pragma: no cover
) -> Callable[[SessionRecord], bool]:  # pragma: no cover
    """Compile ANDed session filters into one predicate."""  # pragma: no cover
    clauses: list[Callable[[SessionRecord], bool]] = []  # pragma: no cover
    for f in filters:  # pragma: no cover
        kind = f.get("kind")  # pragma: no cover
        if kind == "id":  # pragma: no cover
            values = list(f.get("values", []))  # pragma: no cover
            if not values:  # pragma: no cover
                raise invalid_filter(
                    "id filter values must not be empty"
                )  # pragma: no cover

            # pragma: no cover
            def make_id(
                values: list[Any],
            ) -> Callable[[SessionRecord], bool]:  # pragma: no cover
                def match(rec: SessionRecord) -> bool:  # pragma: no cover
                    return _list_matches(
                        rec["header"]["id"], values
                    )  # pragma: no cover

                # pragma: no cover
                return match  # pragma: no cover

            # pragma: no cover
            clauses.append(make_id(values))  # pragma: no cover
        elif kind == "cwd":  # pragma: no cover
            values = list(f.get("values", []))  # pragma: no cover

            # pragma: no cover
            def make_cwd(
                values: list[Any],
            ) -> Callable[[SessionRecord], bool]:  # pragma: no cover
                def match(rec: SessionRecord) -> bool:  # pragma: no cover
                    return _list_matches(
                        rec["header"].get("cwd"), values
                    )  # pragma: no cover

                # pragma: no cover
                return match  # pragma: no cover

            # pragma: no cover
            clauses.append(make_cwd(values))  # pragma: no cover
        elif kind == "created-at":  # pragma: no cover
            seq_range: SessionResultRange = cast(  # pragma: no cover
                SessionResultRange,  # pragma: no cover
                {  # pragma: no cover
                    k: cast(dict[str, Any], f)[k]  # pragma: no cover
                    for k in ("gte", "lte")  # pragma: no cover
                    if k in cast(dict[str, Any], f)  # pragma: no cover
                },  # pragma: no cover
            )  # pragma: no cover
            _validate_range("created-at", seq_range)  # pragma: no cover

            # pragma: no cover
            def make_created_at(  # pragma: no cover
                seq_range: SessionResultRange,  # pragma: no cover
            ) -> Callable[[SessionRecord], bool]:  # pragma: no cover
                def match(rec: SessionRecord) -> bool:  # pragma: no cover
                    return _range_matches(
                        rec["header"].get("createdAt"), seq_range
                    )  # pragma: no cover

                # pragma: no cover
                return match  # pragma: no cover

            # pragma: no cover
            clauses.append(make_created_at(seq_range))  # pragma: no cover
        elif kind == "parent":  # pragma: no cover
            values = list(f.get("values", []))  # pragma: no cover

            # pragma: no cover
            def make_parent(
                values: list[Any],
            ) -> Callable[[SessionRecord], bool]:  # pragma: no cover
                def match(rec: SessionRecord) -> bool:  # pragma: no cover
                    return _list_matches(
                        rec["header"].get("parentSession"), values
                    )  # pragma: no cover

                # pragma: no cover
                return match  # pragma: no cover

            # pragma: no cover
            clauses.append(make_parent(values))  # pragma: no cover
        elif kind == "availability":  # pragma: no cover
            values = list(f.get("values", []))  # pragma: no cover

            # pragma: no cover
            def make_availability(  # pragma: no cover
                values: list[Any],  # pragma: no cover
            ) -> Callable[[SessionRecord], bool]:  # pragma: no cover
                def match(rec: SessionRecord) -> bool:  # pragma: no cover
                    live = rec.get("live", False)  # pragma: no cover
                    persisted = rec.get("persisted", False)  # pragma: no cover
                    for v in values:  # pragma: no cover
                        if v == "live" and live:  # pragma: no cover
                            return True  # pragma: no cover
                        if v == "persisted" and persisted:  # pragma: no cover
                            return True  # pragma: no cover
                        if v == "both" and live and persisted:  # pragma: no cover
                            return True  # pragma: no cover
                    return False  # pragma: no cover

                # pragma: no cover
                return match  # pragma: no cover

            # pragma: no cover
            clauses.append(make_availability(values))  # pragma: no cover
        else:  # pragma: no cover
            raise invalid_filter(
                f"unknown session filter kind: {kind!r}"
            )  # pragma: no cover

    # pragma: no cover
    def predicate(rec: SessionRecord) -> bool:  # pragma: no cover
        return all(clause(rec) for clause in clauses)  # pragma: no cover

    # pragma: no cover
    return predicate  # pragma: no cover


# pragma: no cover
# pragma: no cover
def compile_event_filter(  # pragma: no cover
    filters: list[SessionEventResultFilter],  # pragma: no cover
) -> Callable[[dict[str, Any]], bool]:  # pragma: no cover
    """Compile ANDed event filters into one predicate."""  # pragma: no cover
    clauses: list[Callable[[dict[str, Any]], bool]] = []  # pragma: no cover
    for f in filters:  # pragma: no cover
        kind = f.get("kind")  # pragma: no cover
        if kind == "seq":  # pragma: no cover
            seq_range: SessionResultRange = cast(  # pragma: no cover
                SessionResultRange,  # pragma: no cover
                {  # pragma: no cover
                    k: cast(dict[str, Any], f)[k]  # pragma: no cover
                    for k in ("gte", "lte")  # pragma: no cover
                    if k in cast(dict[str, Any], f)  # pragma: no cover
                },  # pragma: no cover
            )  # pragma: no cover
            _validate_range("seq", seq_range)  # pragma: no cover

            # pragma: no cover
            def make_seq(  # pragma: no cover
                seq_range: SessionResultRange,  # pragma: no cover
            ) -> Callable[[dict[str, Any]], bool]:  # pragma: no cover
                def match(rec: dict[str, Any]) -> bool:  # pragma: no cover
                    return _range_matches(rec["seq"], seq_range)  # pragma: no cover

                # pragma: no cover
                return match  # pragma: no cover

            # pragma: no cover
            clauses.append(make_seq(seq_range))  # pragma: no cover
        elif kind == "time":  # pragma: no cover
            time_range: SessionResultRange = cast(  # pragma: no cover
                SessionResultRange,  # pragma: no cover
                {  # pragma: no cover
                    k: cast(dict[str, Any], f)[k]  # pragma: no cover
                    for k in ("gte", "lte")  # pragma: no cover
                    if k in cast(dict[str, Any], f)  # pragma: no cover
                },  # pragma: no cover
            )  # pragma: no cover
            _validate_range("time", time_range)  # pragma: no cover

            # pragma: no cover
            def make_time(  # pragma: no cover
                time_range: SessionResultRange,  # pragma: no cover
            ) -> Callable[[dict[str, Any]], bool]:  # pragma: no cover
                def match(rec: dict[str, Any]) -> bool:  # pragma: no cover
                    return _range_matches(rec["time"], time_range)  # pragma: no cover

                # pragma: no cover
                return match  # pragma: no cover

            # pragma: no cover
            clauses.append(make_time(time_range))  # pragma: no cover
        elif kind == "type":  # pragma: no cover
            values = list(f.get("values") or [])  # pragma: no cover
            if not values:  # pragma: no cover
                raise invalid_filter(
                    "type filter values must not be empty"
                )  # pragma: no cover

            # pragma: no cover
            def make_type(
                values: list[Any],
            ) -> Callable[[dict[str, Any]], bool]:  # pragma: no cover
                def match(rec: dict[str, Any]) -> bool:  # pragma: no cover
                    return _list_matches(rec["type"], values)  # pragma: no cover

                # pragma: no cover
                return match  # pragma: no cover

            # pragma: no cover
            clauses.append(make_type(values))  # pragma: no cover
        elif kind == "surface":  # pragma: no cover
            values = list(f.get("values") or [])  # pragma: no cover

            # pragma: no cover
            def make_surface(  # pragma: no cover
                values: list[Any],  # pragma: no cover
            ) -> Callable[[dict[str, Any]], bool]:  # pragma: no cover
                def match(rec: dict[str, Any]) -> bool:  # pragma: no cover
                    return _list_matches(rec["surface"], values)  # pragma: no cover

                # pragma: no cover
                return match  # pragma: no cover

            # pragma: no cover
            clauses.append(make_surface(values))  # pragma: no cover
        elif kind == "text":  # pragma: no cover
            text = f.get("text")  # pragma: no cover
            if not isinstance(text, str) or not text:  # pragma: no cover
                raise invalid_filter(
                    "text filter must be a non-empty string"
                )  # pragma: no cover
            pattern = re.compile(re.escape(text), re.IGNORECASE)  # pragma: no cover

            # pragma: no cover
            def make_text(  # pragma: no cover
                pattern: re.Pattern[str],  # pragma: no cover
            ) -> Callable[[dict[str, Any]], bool]:  # pragma: no cover
                def match(rec: dict[str, Any]) -> bool:  # pragma: no cover
                    return bool(
                        pattern.search(str(rec.get("text", "")))
                    )  # pragma: no cover

                # pragma: no cover
                return match  # pragma: no cover

            # pragma: no cover
            clauses.append(make_text(pattern))  # pragma: no cover
        else:  # pragma: no cover
            raise invalid_filter(
                f"unknown event filter kind: {kind!r}"
            )  # pragma: no cover

    # pragma: no cover
    def predicate(rec: dict[str, Any]) -> bool:  # pragma: no cover
        return all(clause(rec) for clause in clauses)  # pragma: no cover

    # pragma: no cover
    return predicate  # pragma: no cover


# pragma: no cover
# pragma: no cover
def event_matches_filter(  # pragma: no cover
    event: SessionEvent,  # pragma: no cover
    filters: list[SessionEventResultFilter],  # pragma: no cover
    extracted_text: str | None = None,  # pragma: no cover
) -> bool:  # pragma: no cover
    """Check one event against compiled filters, returning the match."""  # pragma: no cover
    surface: Literal["current", "shadowed", "log-only"] = "current"  # pragma: no cover
    record = _event_document(event, surface, extracted_text)  # pragma: no cover
    return compile_event_filter(filters)(record)  # pragma: no cover


# pragma: no cover
