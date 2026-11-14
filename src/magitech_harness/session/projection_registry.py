"""Projection registry: eager drive of pure folds over committed events.

One registry per service owns the unit table, the per-session per-unit
watermark cells, the change feed, and the cold-read faces (snapshot,
checkpoint, restore, hydrate). Units are pure synchronous folds; a unit
uninterested in an event returns the same state reference, and the drive
treats that reference identity as "unchanged".
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict, cast
from weakref import WeakKeyDictionary

from anima import Context, Service

from magitech_harness.session.events import SessionEvent

if TYPE_CHECKING:
    from magitech_harness.session.store import Session

#: A validator: takes an unknown value, returns the validated value, and
#: raises ``ValueError`` on invalid input.
Validator = Callable[[Any], Any]


@dataclass(frozen=True)
class WireView[S]:
    """Optional client view. Omit (``wire=None``) for host-only units."""

    view_schema: Validator
    view: Callable[[S], Any]
    summary: bool = True


@dataclass(frozen=True)
class ProjectionDefinition[S]:
    """One domain's pure synchronous fold. Never an opaque getter."""

    key: str
    # Validates persisted state BEFORE it seeds a fold (restore /
    # view_checkpoint), not in checkpoint or apply.
    state_schema: Validator
    # State for the empty log and its immutable session metadata.
    init: Callable[[dict[str, Any], int], S]
    # Pure transition: previous state + one committed event -> next state.
    # A unit uninterested in an event MUST return the same state object.
    apply: Callable[[S, SessionEvent], S]
    wire: WireView[S] | None = None
    # Bump whenever serialized state fields or fold semantics change.
    # Older persisted rows are discarded, never migrated.
    state_version: int = 0


class ProjectionSnapshot(TypedDict):
    """One consistent read cut over every registered client-visible unit."""

    as_of_seq: int
    values: dict[str, Any]


class CheckpointRow(TypedDict):
    """One unit's checkpoint row: its version, watermark, and state."""

    ver: int
    seq: int
    val: Any


@dataclass
class RestoreResult:
    """A cold read: the snapshot cut plus the refreshed checkpoint rows."""

    snapshot: ProjectionSnapshot
    checkpoint: dict[str, CheckpointRow]


@dataclass
class UnitCell:
    """Per-session per-unit watermark cell.

    ``observed_seq`` is the seq of the last event passed through ``apply``
    (regardless of change); ``-1`` for the empty log. ``views`` is the
    fixed two-slot ``[previous_view, current_view]`` buffer for the
    change feed; ``None`` slots mean "no cached comparison".
    """

    state: Any
    observed_seq: int
    views: list[Any]


@dataclass
class _Registration:
    definition: ProjectionDefinition[Any]
    cells: WeakKeyDictionary
    refs: int


def _wire_summary(definition: ProjectionDefinition[Any]) -> bool | None:
    """A unit's summary membership: ``None`` when it has no wire view."""
    if definition.wire is None:
        return None
    return definition.wire.summary


class ProjectionHandle[S]:
    """A registration handle: reads a unit's state, retains it past disposal."""

    def __init__(
        self, registry: SessionProjectionRegistry, registration: _Registration
    ) -> None:
        self._registry = registry
        self._registration = registration
        self._disposed = False

    def state_of(self, session: Session) -> S:
        """The unit's state at the session's current cursor.

        Materializes the cell (lazy full-log fold) if needed. The
        returned state is live; do not mutate it.
        """
        return cast(S, self._registry._cell_for(self._registration, session).state)

    def reader_for(self, session: Session) -> Callable[[], S]:
        """A closure over :meth:`state_of`; retains the unit after disposal."""

        def reader() -> S:
            return self.state_of(session)

        return reader

    def dispose(self) -> None:
        """Release one reference; the last release removes the key."""
        if self._disposed:
            return
        self._disposed = True
        self._registry._release(self._registration)


class SessionProjectionRegistry(Service):
    """The projection unit table and its drive.

    A row service providing ``sessionProjections``. Subscribes once to
    ``session/event``; every committed event passes every registered
    unit's ``apply``. Cells build lazily: a unit registered after events
    flowed, or a session older than the registry, folds ``init`` over
    the in-memory log on first touch.
    """

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx, "sessionProjections")
        self._ctx = ctx
        self._registrations: dict[str, _Registration] = {}
        self._listeners: set[Callable[..., None]] = set()
        self._summary_key_set: frozenset[str] | None = None
        self._disposed = False
        self._created_disposer = ctx.on("session/created", self._on_created)
        self._event_disposer = ctx.on("session/event", self._on_event)

    def dispose(self) -> None:
        """Unregister both listeners; idempotent."""
        if self._disposed:
            return
        self._disposed = True
        for disposer in (self._created_disposer, self._event_disposer):
            if callable(disposer):
                disposer()

    def register[S](self, definition: ProjectionDefinition[S]) -> ProjectionHandle[S]:
        """Install or refcount one unit; returns its handle.

        Raises ``ValueError`` for a non-negative-integer violation, for
        a same-key re-registration at a different ``state_version``, and
        for a same-key re-registration with a different ``summary``.
        """
        version = definition.state_version
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError(
                f'session projection "{definition.key}" state_version must be '
                f"a non-negative integer, got {version!r}"
            )
        existing = self._registrations.get(definition.key)
        if existing is None:
            registration = _Registration(
                definition=definition, cells=WeakKeyDictionary(), refs=1
            )
            self._registrations[definition.key] = registration
            # A new key changes the table; a refcount increment does not.
            self._summary_key_set = None
        else:
            if existing.definition.state_version != version:
                raise ValueError(
                    f'session projection "{definition.key}" is already '
                    f"registered at state_version "
                    f"{existing.definition.state_version}; refusing to share "
                    f"it with state_version {version}"
                )
            if _wire_summary(existing.definition) != _wire_summary(definition):
                raise ValueError(
                    f'session projection "{definition.key}" is already '
                    f"registered with summary="
                    f"{_wire_summary(existing.definition)}; refusing to "
                    f"share it with summary={_wire_summary(definition)}"
                )
            existing.refs += 1
            registration = existing
        return ProjectionHandle(self, registration)

    def _release(self, registration: _Registration) -> None:
        """Drop one reference; the last one out removes the key."""
        registration.refs -= 1
        if registration.refs == 0:
            del self._registrations[registration.definition.key]
            self._summary_key_set = None

    def on_changed(
        self, listener: Callable[[Session, str, Any, int], None]
    ) -> Callable[[], None]:
        """Subscribe to the change feed; returns its disposer.

        The listener is called once per client-visible unit whose raw
        ``view`` result changed by identity, per committed event, with
        the schema-validated value and the seq of the causing event.
        Listeners must be plain synchronous callables.
        """
        self._listeners.add(listener)

        def disposer() -> None:
            self._listeners.discard(listener)

        return disposer

    @staticmethod
    def _selected_keys(
        keys: list[str] | set[str] | frozenset[str] | None,
    ) -> set[str] | frozenset[str] | None:
        """Narrow a key selection: a set is used as given, a list copied."""
        if keys is None:
            return None
        if isinstance(keys, (set, frozenset)):
            return keys
        return set(keys)

    def snapshot(
        self,
        session: Session,
        keys: list[str] | set[str] | frozenset[str] | None = None,
    ) -> ProjectionSnapshot:
        """One consistent cut over every registered client-visible unit.

        Materializes every unit's cell at the session's current cursor
        first, then reads each selected wire unit's view through its
        ``view_schema`` (raises on an invalid view). ``as_of_seq`` is
        the shared log cursor (``-1`` for an empty log). Host-only units
        never appear.
        """
        selected = self._selected_keys(keys)
        self._materialize_cells(session)
        values: dict[str, Any] = {}
        for registration in self._registrations.values():
            definition = registration.definition
            if definition.wire is None:
                continue
            if selected is not None and definition.key not in selected:
                continue
            cell = self._cell_for(registration, session)
            values[definition.key] = self._view_cell(registration, cell)
        return ProjectionSnapshot(as_of_seq=session.seq - 1, values=values)

    def cached_snapshot(
        self,
        session: Session,
        keys: list[str] | set[str] | frozenset[str] | None = None,
    ) -> ProjectionSnapshot | None:
        """Already-materialized cells only: no folding, no materialization.

        Values may trail the live session and are hints, not a complete
        baseline. ``as_of_seq`` is the lowest served watermark; ``None``
        when no wire cell exists.
        """
        selected = self._selected_keys(keys)
        values: dict[str, Any] = {}
        as_of_seq: int | None = None
        for registration in self._registrations.values():
            definition = registration.definition
            if definition.wire is None:
                continue
            if selected is not None and definition.key not in selected:
                continue
            cell = registration.cells.get(session)
            if cell is None:
                continue
            values[definition.key] = self._view_cell(registration, cell)
            if as_of_seq is None or cell.observed_seq < as_of_seq:
                as_of_seq = cell.observed_seq
        if as_of_seq is None:
            return None
        return ProjectionSnapshot(as_of_seq=as_of_seq, values=values)

    def summary_keys(self) -> frozenset[str]:
        """Every registered client-visible key not declared ``summary: False``.

        The returned set is the same instance on every call until a
        registration or its disposal changes the key table.
        """
        if self._summary_key_set is None:
            self._summary_key_set = frozenset(
                registration.definition.key
                for registration in self._registrations.values()
                if registration.definition.wire is not None
                and registration.definition.wire.summary
            )
        return self._summary_key_set

    def checkpoint(self, session: Session) -> dict[str, CheckpointRow]:
        """One row per registered key (host-only units included).

        Each row is detached: the returned ``val`` never aliases the
        live cell.
        """
        rows: dict[str, CheckpointRow] = {}
        for registration in self._registrations.values():
            cell = self._cell_for(registration, session)
            rows[registration.definition.key] = CheckpointRow(
                ver=registration.definition.state_version,
                seq=cell.observed_seq,
                val=copy.deepcopy(cell.state),
            )
        return rows

    def restore_floor(self, checkpoint: dict[str, CheckpointRow]) -> int | None:
        """The stored seq a ``restore`` tail read over ``checkpoint`` starts at.

        One below the lowest usable watermark; a row is usable iff its
        ``ver`` matches the live unit's ``state_version``. An absent or
        mismatched row pulls that key's need to ``0``. ``None`` when no
        unit is registered.
        """
        floor: int | None = None
        for registration in self._registrations.values():
            row = checkpoint.get(registration.definition.key)
            need = 0
            if isinstance(row, dict):
                ver = row.get("ver")
                seq = row.get("seq")
                if (
                    isinstance(ver, int)
                    and not isinstance(ver, bool)
                    and ver == registration.definition.state_version
                    and isinstance(seq, int)
                    and not isinstance(seq, bool)
                ):
                    need = max(seq + 1, 0)
            floor = need if floor is None else min(floor, need)
        if floor is None:
            return None
        return max(floor - 1, 0)

    @staticmethod
    def _row_usable(
        row: Any,
        definition: ProjectionDefinition[Any],
        before_base: int,
        end_seq: int,
    ) -> bool:
        """Whether one checkpoint row may seed a fold over this tail."""
        if not isinstance(row, dict) or "val" not in row:
            return False
        ver = row.get("ver")
        seq = row.get("seq")
        if (
            isinstance(ver, bool)
            or not isinstance(ver, int)
            or ver != definition.state_version
        ):
            return False
        if isinstance(seq, bool) or not isinstance(seq, int):
            return False
        return before_base <= seq <= end_seq

    def restore(
        self,
        checkpoint: dict[str, CheckpointRow],
        events: list[SessionEvent],
        base_seq: int,
        header: dict[str, Any],
        inherited_event_count: int,
    ) -> RestoreResult:
        """Cold read: fold every unit over a stored log suffix.

        Seeds each unit from its checkpoint row when usable, else from
        ``init`` (sound only over the full log, so an unusable row with
        ``base_seq > 0`` raises ``ValueError`` and the caller re-reads
        from seq 0). A row whose persisted state fails ``state_schema``
        raises; an event whose seq breaks contiguity raises. The
        refreshed checkpoint carries the folded live state (not a copy).
        """
        end_seq = events[-1].seq if events else base_seq - 1
        before_base = base_seq - 1
        values: dict[str, Any] = {}
        refreshed: dict[str, CheckpointRow] = {}
        for registration in self._registrations.values():
            definition = registration.definition
            row = checkpoint.get(definition.key)
            usable = self._row_usable(row, definition, before_base, end_seq)
            if not usable and base_seq > 0:
                raise ValueError(
                    f'session projection "{definition.key}" cannot restore '
                    f"from seq {base_seq}: its checkpoint row is missing, "
                    "version-mismatched, or beyond the supplied log end; "
                    "re-read from seq 0"
                )
            if usable:
                assert isinstance(row, dict)
                state = definition.state_schema(row["val"])
                start = row["seq"] - base_seq + 1
            else:
                state = definition.init(header, inherited_event_count)
                start = before_base - base_seq + 1
            for index in range(start, len(events)):
                event = events[index]
                expected = base_seq + index
                if event.seq != expected:
                    raise ValueError(
                        f'session projection "{definition.key}" cannot '
                        f"restore across missing seq {expected}"
                    )
                state = definition.apply(state, event)
            if definition.wire is not None:
                values[definition.key] = definition.wire.view_schema(
                    definition.wire.view(state)
                )
            refreshed[definition.key] = CheckpointRow(
                ver=definition.state_version, seq=end_seq, val=state
            )
        return RestoreResult(
            snapshot=ProjectionSnapshot(as_of_seq=end_seq, values=values),
            checkpoint=refreshed,
        )

    def view_checkpoint(
        self,
        checkpoint: dict[str, CheckpointRow],
        keys: list[str] | set[str] | frozenset[str] | None = None,
    ) -> dict[str, Any]:
        """Zero-I/O view of rows: schema-validated views of stored states.

        Mismatched, malformed, or absent rows leave their key absent;
        each row's ``state_schema`` failure is caught and the row
        skipped (contrast with ``restore``, which raises).
        """
        selected = self._selected_keys(keys)
        values: dict[str, Any] = {}
        for registration in self._registrations.values():
            definition = registration.definition
            if definition.wire is None:
                continue
            if selected is not None and definition.key not in selected:
                continue
            row = checkpoint.get(definition.key)
            if not isinstance(row, dict) or "val" not in row:
                continue
            ver = row.get("ver")
            if (
                isinstance(ver, bool)
                or not isinstance(ver, int)
                or ver != definition.state_version
            ):
                continue
            try:
                state = definition.state_schema(row["val"])
            except Exception:  # noqa: BLE001,S112 - one bad row is skipped, not fatal
                continue
            values[definition.key] = definition.wire.view_schema(
                definition.wire.view(state)
            )
        return values

    def hydrate(
        self,
        session: Session,
        checkpoint: dict[str, CheckpointRow],
        events: list[SessionEvent],
        base_seq: int,
    ) -> ProjectionSnapshot:
        """Restore an exact cut and install its states as live cells.

        Fast path: when every registered unit already has a cell at the
        cut, return the snapshot from the live cells without refolding.
        Otherwise restore, then install each row unless a newer live
        cell already exists for that unit.
        """
        end_seq = events[-1].seq if events else base_seq - 1
        complete = True
        for registration in self._registrations.values():
            cell = registration.cells.get(session)
            if cell is None or cell.observed_seq != end_seq:
                complete = False
                break
        if complete:
            values: dict[str, Any] = {}
            for registration in self._registrations.values():
                definition = registration.definition
                if definition.wire is None:
                    continue
                cell = registration.cells.get(session)
                assert cell is not None
                values[definition.key] = self._view_cell(registration, cell)
            return ProjectionSnapshot(as_of_seq=end_seq, values=values)
        restored = self.restore(
            checkpoint,
            events,
            base_seq,
            session.header,
            session.inherited_event_count,
        )
        for registration in self._registrations.values():
            row = restored.checkpoint[registration.definition.key]
            current = registration.cells.get(session)
            if current is not None and current.observed_seq > row["seq"]:
                continue
            registration.cells[session] = UnitCell(
                state=row["val"], observed_seq=row["seq"], views=[None, None]
            )
        return restored.snapshot

    def _materialize_cells(self, session: Session) -> None:
        """Bring every registered unit's cell to the session's cursor."""
        for registration in self._registrations.values():
            self._cell_for(registration, session)

    def _build_cell(
        self,
        definition: ProjectionDefinition[Any],
        header: dict[str, Any],
        inherited_event_count: int,
        events: list[SessionEvent],
    ) -> UnitCell:
        """Fold one unit from ``init`` over ``events``."""
        state = definition.init(header, inherited_event_count)
        for event in events:
            state = definition.apply(state, event)
        observed = events[-1].seq if events else -1
        return UnitCell(state=state, observed_seq=observed, views=[None, None])

    def _cell_for(self, registration: _Registration, session: Session) -> UnitCell:
        """Read (or lazily build, folding the full in-memory log) one cell."""
        cell = registration.cells.get(session)
        if cell is None:
            cell = self._build_cell(
                registration.definition,
                session.header,
                session.inherited_event_count,
                list(session.events),
            )
            registration.cells[session] = cell
        else:
            self._advance_cell(registration.definition, cell, session, session.seq - 1)
        return cell

    def _advance_cell(
        self,
        definition: ProjectionDefinition[Any],
        cell: UnitCell,
        session: Session,
        through_seq: int,
    ) -> None:
        """Advance one existing cell through a contiguous session prefix."""
        if cell.observed_seq >= through_seq:
            return
        for seq in range(cell.observed_seq + 1, through_seq + 1):
            if seq >= len(session.events) or session.events[seq].seq != seq:
                raise ValueError(
                    f'session projection "{definition.key}" cannot advance '
                    f"across missing seq {seq}"
                )
            event = session.events[seq]
            next_state = definition.apply(cell.state, event)
            if next_state is not cell.state:
                cell.views[0] = cell.views[1]
                cell.views[1] = None
            cell.state = next_state
            cell.observed_seq = seq

    def _on_created(self, session: Session) -> None:
        """Pre-seed one cell per registered unit for a genuinely empty log."""
        if session.seq != 0:
            return
        for registration in self._registrations.values():
            if session in registration.cells:
                continue
            registration.cells[session] = UnitCell(
                state=registration.definition.init(
                    session.header, session.inherited_event_count
                ),
                observed_seq=-1,
                views=[None, None],
            )

    def _on_event(self, session: Session, event: SessionEvent) -> None:
        """Drive one committed event through every registered unit."""
        self._drive(session, event)

    def _drive(self, session: Session, event: SessionEvent) -> None:
        """Eager drive: pass one committed event through every unit."""
        for registration in self._registrations.values():
            definition = registration.definition
            cell = registration.cells.get(session)
            if cell is not None and cell.observed_seq >= event.seq:
                continue
            if cell is None:
                # Late build mid-stream: fold history before this event,
                # then take the normal gate below.
                cell = self._build_cell(
                    definition,
                    session.header,
                    session.inherited_event_count,
                    list(session.events[0 : event.seq]),
                )
                registration.cells[session] = cell
            else:
                self._advance_cell(
                    definition,
                    cell,
                    session,
                    -1 if event.seq == 0 else event.seq - 1,
                )
            previous = cell.state
            next_state = definition.apply(previous, event)
            changed = next_state is not previous
            cell.state = next_state
            cell.observed_seq = event.seq
            wire = definition.wire
            if changed and wire is not None:
                views = cell.views
                views[0] = views[1]
                if self._listeners:
                    views[1] = wire.view(next_state)
                    if views[0] is not views[1]:
                        value = wire.view_schema(views[1])
                        for listener in list(self._listeners):
                            listener(session, definition.key, value, event.seq)
                else:
                    views[1] = None
            # An unchanged state keeps its current view as the valid
            # comparison value for the next state change.

    def _view_cell(self, registration: _Registration, cell: UnitCell) -> Any:
        """One schema-validated wire value."""
        wire = registration.definition.wire
        assert wire is not None
        return wire.view_schema(wire.view(cell.state))
