"""Kernel-rework conformance suite (K6): every kernel claim mapped to its tests.

Each row names one kernel-rework scope item, its status, and the tests
that prove it. A row is ``done`` only when the listed tests exist and
pass; anything else stays listed with its reason instead of being
dropped or quietly reworded. Test node ids are relative to ``tests/``
and resolve to real test functions by static inspection — the map
cannot go stale without this suite failing. The map covers the phase
acceptance proof (section 7 of the kernel plan), the K1–K5 chunk
claims, the K6 scope items, the pinned divergences carried from the
Phase 1 conformance map, and the round-trip gate carried from Phase 2.
"""

import ast
from pathlib import Path

import pytest

# Statuses: done | not-done | deferred | pinned | out-of-scope.
# A row is (id, status, item, tests, note). ``tests`` holds harness node
# ids relative to ``tests/``; rows owned by another repo list no harness
# tests and name their reference in ``note`` instead.
CONFORMANCE_ROWS: tuple[tuple[str, str, str, tuple[str, ...], str], ...] = (
    (
        "K6-ACC1",
        "done",
        "phase acceptance 1: the bundle boot behaves identically to the old direct-mount boot — same nine services, same owned components, same events and logs",
        (
            "unit/magitech_harness/test_kernel_k6.py::test_bundle_boot_exposes_nine_services_and_owned_components",
            "unit/magitech_harness/test_kernel_k6.py::test_cli_commands_end_to_end_on_row_boot",
            "unit/magitech_harness/test_base_bundle.py::test_bundle_slots_are_row_services",
        ),
        "",
    ),
    (
        "K6-ACC2",
        "done",
        "phase acceptance 2: replay of a recorded session reproduces the identical message sequence",
        (
            "unit/magitech_harness/test_kernel_k6.py::test_replay_reproduces_identical_message_sequence",
            "unit/magitech_harness/session/test_phase2_acceptance.py::test_acceptance_replay_recorded_turn_reproduces_identical_sequence",
        ),
        "",
    ),
    (
        "K6-ACC3",
        "done",
        "phase acceptance 3: rename pins against auto-title still hold",
        (
            "unit/magitech_harness/test_kernel_k6.py::test_full_bundle_reversed_still_wires",
            "unit/magitech_harness/session/test_phase2_acceptance.py::test_acceptance_rename_pins_against_auto_title",
            "unit/magitech_harness/session/test_title.py::test_rename_pins_against_automatic_titling",
            "integration/test_title_roundtrip.py::test_title_survives_restart_and_rename_pins",
        ),
        "",
    ),
    (
        "K6-ACC4",
        "done",
        "phase acceptance 4: checkpoint rejection still prevents adapter dispatch",
        (
            "unit/magitech_harness/session/test_checkpoint_policy.py::test_llm_stream_failing_flush_never_dispatches",
            "unit/magitech_harness/session/test_checkpoint_policy.py::test_tool_execute_failing_flush_never_runs_body",
            "unit/magitech_harness/session/test_checkpoint_policy.py::test_pre_step_failing_flush_never_enters_step",
            "unit/magitech_harness/session/test_phase2_acceptance.py::test_acceptance_failing_flush_blocks_llm_and_tool_dispatch",
        ),
        "",
    ),
    (
        "K6-ACC5",
        "done",
        "phase acceptance 5: corrupt / unknown-version files are still refused, not silently repaired",
        (
            "unit/magitech_harness/session/test_phase2_acceptance.py::test_acceptance_corrupt_row_refused_not_repaired",
            "unit/magitech_harness/session/test_phase2_acceptance.py::test_acceptance_unknown_version_refused_without_decoding_rows",
            "unit/magitech_harness/session/test_phase2_acceptance.py::test_acceptance_unknown_non_ignorable_type_refused_at_load",
        ),
        "",
    ),
    (
        "K6-ACC6",
        "done",
        "phase acceptance 6: zero direct mounts remain in create_harness; the kernel is the bare context, the loader, and the bundle",
        (
            "unit/magitech_harness/test_kernel_k6.py::test_create_harness_mounts_nothing_directly",
            "unit/magitech_harness/session/test_kernel_k5.py::test_create_harness_mounts_nothing_directly",
        ),
        "",
    ),
    (
        "K6-ACC7",
        "done",
        "phase acceptance 7: the bundle inventory is pinned — every row, target, and config asserted by test",
        ("unit/magitech_harness/test_base_bundle.py::test_base_bundle_rows",),
        "",
    ),
    (
        "K6-K1",
        "done",
        "K1 leaf rows as bundle rows: the six slots reachable, audit on the sync and running-loop paths",
        (
            "unit/magitech_harness/test_base_bundle.py::test_bundle_slots_are_row_services",
            "unit/magitech_harness/test_base_bundle.py::test_bundle_slots_settle_on_flush_inside_running_loop",
            "unit/magitech_harness/test_base_bundle.py::test_check_bundle_entries_passes_after_flush",
            "unit/magitech_harness/test_base_bundle.py::test_create_harness_fails_loud_on_unresolvable_row",
            "unit/magitech_harness/test_base_bundle.py::test_create_harness_fails_loud_on_failing_row",
        ),
        "",
    ),
    (
        "K6-K2",
        "done",
        "K2 sessions + agent-loop rows: turns complete, titles fire through the loop, replay proof re-run",
        (
            "unit/magitech_harness/session/test_kernel_k5.py::test_automatic_title_fires_through_run_turn",
            "unit/magitech_harness/session/test_kernel_k5.py::test_automatic_title_fires_through_stream_turn",
            "unit/magitech_harness/session/test_kernel_k3.py::test_swapped_rows_still_wire_the_registry",
            "unit/magitech_harness/test_base_bundle.py::test_sessions_row_receives_storage_dir",
        ),
        "",
    ),
    (
        "K6-K3",
        "done",
        "K3 session internals as rows: the five units fold, checkpoint and title proofs re-run, nothing inside the sessions service constructs the moved pieces",
        (
            "unit/magitech_harness/session/test_kernel_k3.py::test_rows_fold_the_five_units",
            "unit/magitech_harness/session/test_kernel_k3.py::test_title_row_config_fails_loud_at_boot",
            "unit/magitech_harness/session/test_kernel_k3.py::test_service_constructs_none_of_the_moved_pieces",
            "unit/magitech_harness/session/test_checkpoint_policy.py::test_llm_stream_flushes_before_first_chunk",
            "unit/magitech_harness/session/test_checkpoint_policy.py::test_tool_execute_flushes_before_top_level_body",
            "unit/magitech_harness/session/test_checkpoint_policy.py::test_pre_step_flushes_before_step",
            "unit/magitech_harness/session/test_title.py::test_rename_pins_against_automatic_titling",
            "unit/magitech_harness/session/test_title.py::test_refresh_unpins_user_title_without_provider",
        ),
        "",
    ),
    (
        "K6-K4",
        "done",
        "K4 persistence + projection-cache rows: torn-tail and cache proofs re-run, memory-only boot stays hollow",
        (
            "unit/magitech_harness/session/test_kernel_k4.py::test_swapped_cache_row_still_wires",
            "unit/magitech_harness/session/test_kernel_k4.py::test_bundle_slots_with_storage",
            "unit/magitech_harness/session/test_kernel_k4.py::test_memory_only_boot_rows_are_hollow",
            "unit/magitech_harness/session/test_kernel_k4.py::test_bundle_without_persistence_row_fails_loud",
            "unit/magitech_harness/session/test_kernel_k4.py::test_persistence_row_with_root_installs_tracker",
            "unit/magitech_harness/session/test_kernel_k4.py::test_cache_row_with_root_installs_cache",
            "unit/magitech_harness/session/test_persistence_jsonl.py::test_handle_append_repairs_torn_tail",
            "unit/magitech_harness/session/test_persistence_jsonl.py::test_lease_second_acquire_in_process_is_owned",
            "integration/test_projection_cache_roundtrip.py::test_kill_at_put_cache_never_ahead_of_log",
        ),
        "",
    ),
    (
        "K6-K5",
        "done",
        "K5 provider + components rows: live titles from the provider row, components settle on a root flush, disabling the provider row restores the fallback",
        (
            "unit/magitech_harness/session/test_kernel_k5.py::test_automatic_title_fires_through_run_turn",
            "unit/magitech_harness/session/test_kernel_k5.py::test_automatic_title_fires_through_stream_turn",
            "unit/magitech_harness/session/test_kernel_k5.py::test_row_mounted_components_settle_on_root_flush",
            "unit/magitech_harness/session/test_kernel_k5.py::test_disabling_provider_row_restores_fallback",
            "unit/magitech_harness/session/test_kernel_k5.py::test_swapped_new_rows_still_wire",
        ),
        "",
    ),
    (
        "K6-SWAP",
        "done",
        "swapped-row probes consolidated: the whole bundle reversed still wires, order carries no load semantics",
        (
            "unit/magitech_harness/test_kernel_k6.py::test_full_bundle_reversed_still_wires",
            "unit/magitech_harness/session/test_kernel_k3.py::test_swapped_rows_still_wire_the_registry",
            "unit/magitech_harness/session/test_kernel_k4.py::test_swapped_cache_row_still_wires",
            "unit/magitech_harness/session/test_kernel_k5.py::test_swapped_new_rows_still_wire",
        ),
        "",
    ),
    (
        "K6-CHILD",
        "done",
        "child-turn proof: a child context built from the bundle runs a turn and fires its own provider title, leaving the parent untouched",
        (
            "unit/magitech_harness/test_kernel_k6.py::test_child_harness_from_bundle_runs_a_turn",
        ),
        "",
    ),
    (
        "K6-CLI",
        "done",
        "CLI headless / repl / session list / session show / component list driven end to end on the row boot; verb-first run dropped",
        (
            "unit/magitech_harness/test_kernel_k6.py::test_cli_commands_end_to_end_on_row_boot",
            "unit/magitech_harness/test_cli.py::test_main_async_run_subcommand_removed",
            "unit/magitech_harness/test_cli.py::test_main_async_session_commands",
            "unit/magitech_harness/test_cli.py::test_main_async_component_command",
        ),
        "",
    ),
    (
        "K6-DISP",
        "done",
        "disposal proof: disabling a row cleanly unmounts its slot and listeners, by disable and by reload without the row",
        (
            "unit/magitech_harness/test_kernel_k6.py::test_disabling_row_unmounts_slot_and_listeners",
            "unit/magitech_harness/test_kernel_k6.py::test_disabling_shell_row_unmounts_slot",
            "unit/magitech_harness/test_kernel_k6.py::test_disabling_policy_row_removes_checkpoint_listeners",
        ),
        "",
    ),
    (
        "K6-DISP-A",
        "done",
        "bad cache throttles + no registry: no raise — the bare service builds no cache, so the values are never validated there",
        (
            "unit/magitech_harness/test_kernel_k6.py::test_bare_service_with_bad_cache_throttles_does_not_raise",
            "unit/magitech_harness/session/test_projection_cache.py::test_config_rejects_bad_write_every_events",
        ),
        "",
    ),
    (
        "K6-DISP-B",
        "done",
        "unit-row unload while the title service holds the key: the fallback title still lands",
        (
            "unit/magitech_harness/test_kernel_k6.py::test_unit_row_unload_keeps_title_via_title_service",
        ),
        "",
    ),
    (
        "K6-DISP-C",
        "done",
        "title-row unload fires dispose through the registered effect: the held service reports disposed, no new title lands, dependents stay pending and the audit names them",
        (
            "unit/magitech_harness/test_kernel_k6.py::test_title_row_unload_disposes_service_via_effect",
        ),
        "",
    ),
    (
        "K6-AUDIT",
        "done",
        "boot-time slot assertion: the audit fails loud on a bad row on the sync path and after flush on the running-loop path",
        (
            "unit/magitech_harness/test_base_bundle.py::test_check_bundle_entries_fails_loud_on_unresolvable_row_after_flush",
            "unit/magitech_harness/test_base_bundle.py::test_check_bundle_entries_fails_loud_on_failing_row_after_flush",
            "unit/magitech_harness/test_base_bundle.py::test_create_harness_fails_loud_on_unresolvable_row",
        ),
        "",
    ),
    (
        "K6-N1",
        "done",
        "audit fails loud before flush on the running-loop path when zero loader entries exist",
        (
            "unit/magitech_harness/test_kernel_k6.py::test_check_bundle_entries_fails_before_flush_on_loop_path",
        ),
        "",
    ),
    (
        "K6-CLI-TITLE",
        "pinned",
        "CLI exits without draining deferred title work, so no title event lands in CLI-driven sessions before exit",
        (),
        "Known behaviour, identical on main, pre-existing: the CLI returns without draining the title service's deferred work. Flagged for Phase 3; not fixed here.",
    ),
    (
        "K6-K0",
        "done",
        "loader-probe conclusions the row shape relies on, re-proved as tests: mirror visibility and timing, mirror satisfies inject, direct mount wins, env interpolation, disposal, store ctx",
        (
            "unit/magitech_harness/test_kernel_k6.py::test_check_bundle_entries_fails_before_flush_on_loop_path",
            "unit/magitech_harness/test_kernel_k6.py::test_check_bundle_entries_sync_path_settles_at_load",
            "unit/magitech_harness/test_kernel_k6.py::test_full_bundle_reversed_still_wires",
            "unit/magitech_harness/test_kernel_k6.py::test_direct_mount_wins_over_bundle_row",
            "unit/magitech_harness/test_kernel_k6.py::test_disabling_row_unmounts_slot_and_listeners",
            "unit/magitech_harness/test_base_bundle.py::test_sessions_row_receives_storage_dir",
            "unit/magitech_harness/test_kernel_k6.py::test_replay_reproduces_identical_message_sequence",
        ),
        "",
    ),
    (
        "K6-OQ3",
        "done",
        "kept main_async boot shape: bundle rows, then the user profile, then the mock fallback, then dispatch",
        (
            "unit/magitech_harness/test_cli.py::test_main_async_boot_order_bundle_profile_fallback_dispatch",
            "unit/magitech_harness/test_cli.py::test_main_async_mock_fallback_runs_before_dispatch",
            "unit/magitech_harness/test_cli.py::test_main_async_profile_and_default_repl",
        ),
        "",
    ),
    (
        "K6-OQ5",
        "deferred",
        "no storage-domain stack for the projection cache: the cache writes direct to the storage dir",
        (
            "unit/magitech_harness/session/test_kernel_k4.py::test_cache_row_with_root_installs_cache",
        ),
        "Decided deferral: the storage-domain split is a larger refactor, deferred to a later phase.",
    ),
    (
        "K6-NOAGENT",
        "out-of-scope",
        "title service does not inject an agent registry: none exists yet",
        (),
        "Deliberate simplification recorded in K3: the row injects only what exists (sessions, sessionProjections).",
    ),
    (
        "K6-NONEW",
        "out-of-scope",
        "no rows beyond existing functionality: telemetry, web, MCP, skills, goals, subagent providers, and the rest",
        (),
        "Explicitly out of the kernel-rework scope; later-phase plugin work, not this phase.",
    ),
    (
        "K6-ACC06",
        "done",
        "bidirectional v4 wire session and projection checkpoint round-trip",
        (
            "integration/test_cross_tooling_roundtrip.py::test_bidirectional_v4_session_roundtrip",
            "integration/test_cross_tooling_roundtrip.py::test_bidirectional_v4_catalog_team_and_projection_roundtrip",
        ),
        "",
    ),
    (
        "PIN-01",
        "pinned",
        "parallel dispatch raises the first error instead of an aggregate",
        (),
        "Pinned in anima tests/integration/test_phase1_acceptance.py::test_acceptance_dispatch_modes. Needs a design decision, not a drive-by fix.",
    ),
    (
        "PIN-02",
        "pinned",
        "context provide notifies at registration time rather than after ACTIVE",
        (),
        "Pinned in anima tests/integration/test_phase1_acceptance.py::test_acceptance_mount_event_sequence (count-only). Needs a design decision, not a drive-by fix.",
    ),
)

# The complete inventory, stated independently of the table above. A row
# may not be added or removed without updating this set in the same
# change, so coverage cannot shrink silently.
EXPECTED_ROW_IDS: frozenset[str] = frozenset(
    {
        "K6-ACC1",
        "K6-ACC2",
        "K6-ACC3",
        "K6-ACC4",
        "K6-ACC5",
        "K6-ACC6",
        "K6-ACC7",
        "K6-K1",
        "K6-K2",
        "K6-K3",
        "K6-K4",
        "K6-K5",
        "K6-SWAP",
        "K6-CHILD",
        "K6-CLI",
        "K6-DISP",
        "K6-DISP-A",
        "K6-DISP-B",
        "K6-DISP-C",
        "K6-AUDIT",
        "K6-N1",
        "K6-CLI-TITLE",
        "K6-K0",
        "K6-OQ3",
        "K6-OQ5",
        "K6-NOAGENT",
        "K6-NONEW",
        "K6-ACC06",
        "PIN-01",
        "PIN-02",
    }
)

_TESTS_ROOT = Path(__file__).parents[3]


def _defined_test_names(path: Path) -> set[str]:
    """Every test function defined in one test file, by static inspection."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name.startswith("test_")
        ):
            names.add(node.name)
    return names


def _resolve(node_id: str) -> tuple[Path, str]:
    """Split a node id into its file and bare test name.

    Node ids are relative to ``tests/``: ``unit/...`` or
    ``integration/...``.
    """
    relative, name = node_id.split("::")
    name = name.split("[", 1)[0]
    return _TESTS_ROOT / relative, name


def _listed_nodes() -> list[tuple[str, str]]:
    """Every (row id, node id) pair in the table."""
    return [(row[0], node) for row in CONFORMANCE_ROWS for node in row[3]]


@pytest.mark.parametrize(
    ("row_id", "node_id"), _listed_nodes(), ids=[f"{r}:{n}" for r, n in _listed_nodes()]
)
def test_conformance_listed_test_exists(row_id: str, node_id: str) -> None:
    """Every test a row cites must exist, or the map is stale."""
    path, name = _resolve(node_id)
    assert path.is_file(), f"{row_id}: {node_id} names a missing file"
    assert name in _defined_test_names(path), (
        f"{row_id}: {node_id} names a missing test"
    )


def test_conformance_done_rows_have_tests() -> None:
    """A row may claim done only when it names the tests that prove it.

    A done row owned by another repo names that repo's proof in its note
    instead of listing harness tests.
    """
    missing = [
        row[0]
        for row in CONFORMANCE_ROWS
        if row[1] == "done" and not row[3] and "repo" not in row[4]
    ]
    assert missing == [], f"done rows without tests: {missing}"


def test_conformance_non_done_rows_carry_a_reason() -> None:
    """Every row that is not done must say why, in its own note."""
    silent = [
        row[0] for row in CONFORMANCE_ROWS if row[1] != "done" and not row[4].strip()
    ]
    assert silent == [], f"non-done rows without a reason: {silent}"


def test_conformance_row_ids_are_unique_and_complete() -> None:
    """Row ids are unique, and the inventory is exactly the expected set."""
    ids = [row[0] for row in CONFORMANCE_ROWS]
    assert len(ids) == len(set(ids)), "duplicate conformance row ids"
    assert frozenset(ids) == EXPECTED_ROW_IDS, (
        "conformance inventory changed without updating EXPECTED_ROW_IDS"
    )


def test_conformance_bidirectional_round_trip_is_done() -> None:
    """The cross-tooling round-trip conformance row is done and cites real tests."""
    row = next(row for row in CONFORMANCE_ROWS if row[0] == "K6-ACC06")
    assert row[1] == "done"
    assert len(row[3]) >= 1


def test_conformance_pinned_divergences_stay_pinned() -> None:
    """The pinned divergences stay pinned, not fixed and not dropped."""
    pinned = {row[0]: row for row in CONFORMANCE_ROWS if row[1] == "pinned"}
    assert set(pinned) == {"PIN-01", "PIN-02", "K6-CLI-TITLE"}
    for row in pinned.values():
        assert row[4].strip()
    assert "design decision" in pinned["PIN-01"][4]
    assert "design decision" in pinned["PIN-02"][4]
    assert "Phase 3" in pinned["K6-CLI-TITLE"][4]


def test_conformance_pre_flush_audit_guard_is_done() -> None:
    """The pre-flush zero-entry audit guard is now done and proved."""
    row = next(row for row in CONFORMANCE_ROWS if row[0] == "K6-N1")
    assert row[1] == "done"
    assert (
        "unit/magitech_harness/test_kernel_k6.py::test_check_bundle_entries_fails_before_flush_on_loop_path"
        in row[3]
    )
