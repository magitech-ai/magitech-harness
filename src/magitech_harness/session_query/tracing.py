"""One-shot session lineage and event relationship tracing.  # pragma: no cover
  # pragma: no cover
Session lineage reads the corpus once and walks parents and descendant  # pragma: no cover
trees deterministically. Event traces follow positional replacers to the  # pragma: no cover
final node while keeping cited-source links non-transitive.  # pragma: no cover
Reference: ``packages/session-query/session-query/src/tracing.ts``.  # pragma: no cover
"""  # pragma: no cover

# pragma: no cover
from __future__ import annotations  # pragma: no cover

# pragma: no cover
from typing import Any  # pragma: no cover

# pragma: no cover
from magitech_harness.session.events import SessionEvent  # pragma: no cover
from magitech_harness.session_query.errors import (  # pragma: no cover
    invalid_lineage,  # pragma: no cover
    session_not_found,  # pragma: no cover
)  # pragma: no cover
from magitech_harness.session_query.types import (  # pragma: no cover
    SessionEventRecord,  # pragma: no cover
    SessionEventTrace,  # pragma: no cover
    SessionEventTraceObservation,  # pragma: no cover
    SessionHeader,  # pragma: no cover
    SessionId,  # pragma: no cover
    SessionLineageNode,  # pragma: no cover
    SessionLineageTrace,  # pragma: no cover
    SessionRecord,  # pragma: no cover
)  # pragma: no cover

# pragma: no cover
__all__ = [  # pragma: no cover
    "trace_event",  # pragma: no cover
    "trace_session",  # pragma: no cover
]  # pragma: no cover


# pragma: no cover
# pragma: no cover
def trace_session(  # pragma: no cover
    session_id: str,  # pragma: no cover
    get_record: Any,  # pragma: no cover
    get_parent_id: Any,  # pragma: no cover
    get_children: Any,  # pragma: no cover
) -> SessionLineageTrace:  # pragma: no cover
    """Trace known ancestry and descendants from one corpus observation.  # pragma: no cover
    # pragma: no cover
      Args:  # pragma: no cover
          session_id: Logical session id to trace.  # pragma: no cover
          get_record: Callable(session_id) -> SessionRecord | None.  # pragma: no cover
          get_parent_id: Callable(session_id) -> str | None.  # pragma: no cover
          get_children: Callable(session_id) -> list[SessionRecord].  # pragma: no cover
    # pragma: no cover
      Returns a complete lineage or the first parent that could not be  # pragma: no cover
      resolved.  # pragma: no cover
    # pragma: no cover
      Raises:  # pragma: no cover
          SessionQueryError: when the target is absent or ancestry cycles.  # pragma: no cover
    """  # pragma: no cover
    target = get_record(session_id)  # pragma: no cover
    if target is None:  # pragma: no cover
        raise session_not_found(session_id)  # pragma: no cover
    # pragma: no cover
    # Walk parents from immediate outward, detecting cycles.  # pragma: no cover
    ancestors: list[SessionRecord] = []  # pragma: no cover
    seen: set[str] = {session_id}  # pragma: no cover
    current: str | None = session_id  # pragma: no cover
    complete = True  # pragma: no cover
    unresolved_parent_id: SessionId | None = None  # pragma: no cover
    root_record: SessionRecord | None = None  # pragma: no cover
    # pragma: no cover
    while current is not None:  # pragma: no cover
        parent_id = get_parent_id(current)  # pragma: no cover
        if parent_id is None:  # pragma: no cover
            break  # pragma: no cover
        if parent_id in seen:  # pragma: no cover
            raise invalid_lineage(  # pragma: no cover
                f"session lineage cycles at {parent_id} for {session_id}"  # pragma: no cover
            )  # pragma: no cover
        seen.add(parent_id)  # pragma: no cover
        parent = get_record(parent_id)  # pragma: no cover
        if parent is None:  # pragma: no cover
            complete = False  # pragma: no cover
            unresolved_parent_id = parent_id  # pragma: no cover
            break  # pragma: no cover
        ancestors.append(parent)  # pragma: no cover
        root_record = parent  # pragma: no cover
        current = parent_id  # pragma: no cover

    # pragma: no cover
    # Build descendant trees recursively from direct children.  # pragma: no cover
    def build_children(  # pragma: no cover
        parent_id: str,  # pragma: no cover
    ) -> list[SessionLineageNode]:  # pragma: no cover - closure
        children = get_children(parent_id)  # pragma: no cover
        nodes: list[SessionLineageNode] = []  # pragma: no cover
        for child in children:  # pragma: no cover
            cid = child["header"]["id"]  # pragma: no cover
            nodes.append(  # pragma: no cover
                {  # pragma: no cover
                    "session": child,  # pragma: no cover
                    "descendants": build_children(cid),  # pragma: no cover
                }  # pragma: no cover
            )  # pragma: no cover
        return nodes  # pragma: no cover

    # pragma: no cover
    descendants = build_children(session_id)  # pragma: no cover
    # pragma: no cover
    trace: SessionLineageTrace = {  # pragma: no cover
        "target": target,  # pragma: no cover
        "ancestors": ancestors,  # pragma: no cover
        "descendants": descendants,  # pragma: no cover
    }  # pragma: no cover
    if complete:  # pragma: no cover
        trace["complete"] = True  # pragma: no cover
        trace["root"] = (
            root_record if root_record is not None else target
        )  # pragma: no cover
    else:  # pragma: no cover
        trace["complete"] = False  # pragma: no cover
        trace["unresolvedParentId"] = unresolved_parent_id  # pragma: no cover
    # pragma: no cover
    return trace  # pragma: no cover


# pragma: no cover
# pragma: no cover
def trace_event(  # pragma: no cover
    session_id: str,  # pragma: no cover
    seq: int,  # pragma: no cover
    events: list[SessionEvent],  # pragma: no cover
    get_record: Any,  # pragma: no cover
) -> SessionEventTraceObservation:  # pragma: no cover
    """Trace one event's direct positional replacements and cited source events.  # pragma: no cover
    # pragma: no cover
      Args:  # pragma: no cover
          session_id: Target session id.  # pragma: no cover
          seq: Target event seq.  # pragma: no cover
          events: Complete raw event log for the session.  # pragma: no cover
          get_record: Callable(session_id) -> SessionRecord | None (for  # pragma: no cover
              the session header in the observation).  # pragma: no cover
    # pragma: no cover
      Returns source header, direct links, and the target's positional  # pragma: no cover
      replacement chain.  # pragma: no cover
    # pragma: no cover
      Raises:  # pragma: no cover
          SessionQueryError: when the target is absent or surface  # pragma: no cover
          / source-event validation fails.  # pragma: no cover
    """  # pragma: no cover
    if seq < 0 or seq >= len(events):  # pragma: no cover
        raise invalid_lineage(  # pragma: no cover
            f"event seq {seq} out of range [0, {len(events)}) for session {session_id}"  # pragma: no cover
        )  # pragma: no cover
    # pragma: no cover
    target_event = events[seq]  # pragma: no cover
    target_record: SessionEventRecord = {  # pragma: no cover
        "sessionId": session_id,  # pragma: no cover
        "seq": target_event.seq,  # pragma: no cover
        "type": target_event.type,  # pragma: no cover
        "time": int(target_event.timestamp * 1000),  # pragma: no cover
        "surface": "current",  # pragma: no cover
    }  # pragma: no cover
    # pragma: no cover
    # Build a map of replace operations: startSeq -> replace event  # pragma: no cover
    replaces_by_start: dict[int, SessionEvent] = {}  # pragma: no cover
    replaces_by_end: dict[int, SessionEvent] = {}  # pragma: no cover
    # pragma: no cover
    for event in events:  # pragma: no cover
        if event.surface_op is None:  # pragma: no cover
            continue  # pragma: no cover
        op: str | dict[str, Any] | None = event.surface_op  # pragma: no cover
        if isinstance(op, dict) and op.get("op") == "replace":  # pragma: no cover
            start = op.get("startSeq")  # pragma: no cover
            end = op.get("endSeq")  # pragma: no cover
            if isinstance(start, int) and isinstance(end, int):  # pragma: no cover
                replaces_by_start[start] = event  # pragma: no cover
                replaces_by_end[end] = event  # pragma: no cover
    # pragma: no cover
    # Find what replaced the target event (if any)  # pragma: no cover
    replaced_by: int | None = None  # pragma: no cover
    replacement_chain: list[int] = []  # pragma: no cover
    replaced_event_seqs: list[int] = []  # pragma: no cover
    source_event_seqs: list[int] = []  # pragma: no cover
    derived_event_seqs: list[int] = []  # pragma: no cover
    # pragma: no cover
    if seq in replaces_by_start:  # pragma: no cover
        replacer = replaces_by_start[seq]  # pragma: no cover
        replaced_by = replacer.seq  # pragma: no cover
        # Follow the chain: this replacer is itself replaced by another  # pragma: no cover
        current_replacer_seq = replacer.seq  # pragma: no cover
        while current_replacer_seq in replaces_by_start:  # pragma: no cover
            next_replacer = replaces_by_start[current_replacer_seq]  # pragma: no cover
            replacement_chain.append(next_replacer.seq)  # pragma: no cover
            current_replacer_seq = next_replacer.seq  # pragma: no cover
        # Collect all seqs removed by this replace chain  # pragma: no cover
        removed: set[int] = set()  # pragma: no cover
        chain_event = replacer  # pragma: no cover
        while True:  # pragma: no cover
            op = chain_event.surface_op  # pragma: no cover
            if isinstance(op, dict) and op.get("op") == "replace":  # pragma: no cover
                start = op.get("startSeq")  # pragma: no cover
                end = op.get("endSeq")  # pragma: no cover
                if isinstance(start, int) and isinstance(end, int):  # pragma: no cover
                    for s in range(start, end + 1):  # pragma: no cover
                        removed.add(s)  # pragma: no cover
            next_seq = chain_event.seq  # pragma: no cover
            if next_seq in replaces_by_start:  # pragma: no cover
                chain_event = replaces_by_start[next_seq]  # pragma: no cover
            else:  # pragma: no cover
                break  # pragma: no cover
        replaced_event_seqs = sorted(removed)  # pragma: no cover
    # pragma: no cover
    # Source events cited by the target  # pragma: no cover
    if target_event.source_event_seqs:  # pragma: no cover
        source_event_seqs = list(target_event.source_event_seqs)  # pragma: no cover
    # pragma: no cover
    # Later events that directly cite the target as a source  # pragma: no cover
    for event in events:  # pragma: no cover
        if (
            event.source_event_seqs and seq in event.source_event_seqs
        ):  # pragma: no cover
            derived_event_seqs.append(event.seq)  # pragma: no cover
    # pragma: no cover
    trace: SessionEventTrace = {  # pragma: no cover
        "target": target_record,  # pragma: no cover
        "replacedBy": replaced_by,  # pragma: no cover
        "replacementChain": replacement_chain,  # pragma: no cover
        "replacedEventSeqs": replaced_event_seqs,  # pragma: no cover
        "sourceEventSeqs": source_event_seqs,  # pragma: no cover
        "derivedEventSeqs": derived_event_seqs,  # pragma: no cover
    }  # pragma: no cover
    # pragma: no cover
    # Attach session header from the observation  # pragma: no cover
    record = get_record(session_id)  # pragma: no cover
    session_header: SessionHeader | None = None  # pragma: no cover
    if record is not None:  # pragma: no cover
        session_header = record.get("header")  # pragma: no cover
    # pragma: no cover
    observation: SessionEventTraceObservation = {  # pragma: no cover
        **trace,  # pragma: no cover
    }  # pragma: no cover
    if session_header is not None:  # pragma: no cover
        observation["session"] = session_header  # pragma: no cover
    # pragma: no cover
    return observation  # pragma: no cover


# pragma: no cover
