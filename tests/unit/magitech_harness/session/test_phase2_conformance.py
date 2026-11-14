"""Phase 2 conformance suite (P2-8): every scope item mapped to its tests.

Each row names one Phase 2 scope item, its status, and the tests that
prove it. A row is ``done`` only when the listed tests exist and pass;
anything else stays listed with its reason instead of being dropped or
quietly reworded. Test node ids are relative to ``tests/`` and resolve
to real test functions by static inspection — the map cannot go stale
without this suite failing.
"""

import ast
from pathlib import Path

import pytest

# Statuses: done | not-done | deferred | pinned | out-of-scope.
# A row is (id, status, item, tests, note). ``tests`` holds harness node
# ids; rows owned by another repo list no harness tests and name their
# reference in ``note`` instead.
CONFORMANCE_ROWS: tuple[tuple[str, str, str, tuple[str, ...], str], ...] = (
    (
        "FMT-01",
        "done",
        "format v4, header-only classification; newer refused with upgrade text, malformed as malformed",
        (
            "test_format.py::test_read_header_current",
            "test_format.py::test_read_header_migration_required",
            "test_format.py::test_read_header_newer_version_is_unsupported",
            "test_format.py::test_read_header_malformed",
            "test_format.py::test_session_format_version_refusal_text",
            "test_phase2_acceptance.py::test_acceptance_unknown_version_refused_without_decoding_rows",
        ),
        "",
    ),
    (
        "FMT-02",
        "done",
        "frozen adjacent-only migrations v0 to v1 to v2 to v3 to v4, byte-equivalent v4 output",
        (
            "test_format.py::test_e2e_v0_to_v4",
            "test_format.py::test_e2e_v1_to_v4",
            "test_format.py::test_e2e_v2_to_v4",
            "test_format.py::test_e2e_v3_to_v4",
            "test_phase2_acceptance.py::test_acceptance_valid_v2_chronology_migrates_and_replays",
            "test_phase2_acceptance.py::test_acceptance_v2_and_v4_generations_replay_identical_sequence",
        ),
        "",
    ),
    (
        "FMT-03",
        "done",
        "catalog rule: exactly one codec per version, enforced at construction",
        (
            "test_format.py::test_catalog_duplicate_codec_throws",
            "test_format.py::test_catalog_missing_codec_throws",
            "test_format.py::test_catalog_newer_codec_throws",
            "test_format.py::test_catalog_missing_migration_throws",
            "test_format.py::test_catalog_duplicate_migration_throws",
        ),
        "",
    ),
    (
        "FMT-04",
        "done",
        "seq-range encoding of stored rows",
        (
            "test_format.py::test_encode_seq_ranges_round_trip",
            "test_format.py::test_decode_seq_ranges_rejects_bad_ranges",
        ),
        "",
    ),
    (
        "FMT-05",
        "done",
        "inherited cut recovered from the tagged end-seed marker, not the header",
        (
            "test_format.py::test_v1_to_v2_seeded_marker_tagged_and_cut_recovered",
            "test_format.py::test_v2_to_v3_seeded_cut_recovered",
            "test_format.py::test_e2e_seeded_cut_from_end_seed_marker",
        ),
        "",
    ),
    (
        "FMT-06",
        "done",
        "delivery-acceptance watermark fold",
        (
            "test_format.py::test_accepted_through_folds_matching_markers",
            "test_format.py::test_accepted_through_rejects_malformed_watermark",
            "test_format.py::test_validate_delivery_accepted",
        ),
        "",
    ),
    (
        "EVT-01",
        "done",
        "62-type event registry reproducing the owner merge",
        (
            "test_events.py::test_registry_is_exactly_62_types",
            "test_events.py::test_registry_reproduces_the_owner_merge",
            "test_events.py::test_event_type_covers_known_session_event_types",
        ),
        "",
    ),
    (
        "EVT-02",
        "done",
        "seq assign-on-append, verify-on-load, reject gaps and duplicates",
        (
            "test_store.py::test_session_appends_all_62_types_and_replays_in_order",
            "test_store.py::test_session_load_refuses_gapped_seq",
            "test_store.py::test_session_load_refuses_duplicated_seq",
        ),
        "",
    ),
    (
        "EVT-03",
        "done",
        "unknown non-ignorable event refused at restore, storage validation, and scan; ignorable carried opaquely",
        (
            "test_store.py::test_unknown_non_ignorable_refused_at_all_three_checkpoints",
            "test_store.py::test_session_load_carries_ignorable_unknown_and_refuses_known_unknown",
            "test_phase2_acceptance.py::test_acceptance_unknown_non_ignorable_type_refused_at_load",
        ),
        "",
    ),
    (
        "EVT-04",
        "done",
        "retired shapes refused",
        (
            "test_store.py::test_session_load_refuses_retired_shapes",
            "test_store.py::test_session_load_refuses_tool_result_wrapper_blocks",
        ),
        "",
    ),
    (
        "STO-01",
        "done",
        "prepare / enter / announce transaction",
        (
            "test_store.py::test_store_prepare_mints_ids_and_validates_meta",
            "test_store.py::test_store_announce_exactly_once_and_veto",
            "test_store.py::test_store_enter_rechecks_duplicates",
        ),
        "",
    ),
    (
        "STO-02",
        "done",
        "surface fold: append/replace ops, system-head protection, developer headerSeq rules",
        (
            "test_surface.py::test_fold_replace_shadows_range_and_records_replacement",
            "test_surface.py::test_fold_system_head_protection",
            "test_surface.py::test_fold_developer_header_seq_rules",
        ),
        "",
    ),
    (
        "STO-03",
        "done",
        "fork via seed plus synthetic closers; closed steps and turns preserved",
        (
            "test_fork.py::test_build_fork_seed_mid_log_boundary",
            "test_fork.py::test_build_fork_seed_open_turn_gets_synthetic_closers",
            "test_fork.py::test_build_fork_seed_closed_steps_and_turns_preserved",
        ),
        "",
    ),
    (
        "STO-04",
        "done",
        "plugin records always ignorable and owner-validated",
        (
            "test_store.py::test_session_plugin_records",
            "test_store.py::test_session_plugin_record_carried_on_load",
        ),
        "",
    ),
    (
        "PER-01",
        "done",
        "persistence contract and ownership errors; ownership-lost declared but never thrown",
        (
            "test_persistence.py::test_ownership_lost_error_exists_but_is_never_thrown",
            "test_persistence.py::test_open_write_while_owned_raises_already_owned",
            "test_persistence.py::test_create_duplicate_id_raises_already_exists",
        ),
        "",
    ),
    (
        "PER-02",
        "done",
        "JSONL generations resolved by directory listing; kernel lease; torn-tail recovery",
        (
            "test_persistence_jsonl.py::test_resolve_current_generation_highest_wins",
            "test_persistence_jsonl.py::test_lease_second_acquire_in_process_is_owned",
            "test_persistence_jsonl.py::test_handle_append_repairs_torn_tail",
            "test_persistence_jsonl.py::test_scan_log_bytes_complete_and_torn",
        ),
        "",
    ),
    (
        "PER-03",
        "done",
        "migration publication verified in-process before the source generation is replaced",
        (
            "test_persistence_jsonl.py::test_service_open_write_publishes_historical_migration",
            "test_persistence_jsonl.py::test_service_open_read_historical_primed_without_publication",
        ),
        "",
    ),
    (
        "PER-04",
        "done",
        "header-only stat/list with format status; delete as a documented admin operation",
        (
            "test_persistence_jsonl.py::test_service_stat_historical_is_migration_required",
            "test_service.py::test_session_service_load_list_and_delete",
        ),
        "",
    ),
    (
        "PRJ-01",
        "done",
        "projection registry: eager drive, watermark cells, snapshot and restore",
        (
            "test_projection_registry.py::test_hydrate_installs_cells_and_later_drive_advances_once",
            "test_projection_registry.py::test_restore_discards_version_mismatch_at_base_zero",
        ),
        "",
    ),
    (
        "PRJ-02",
        "done",
        "shipped projections: title, title input, session stats, turn outline, working directory",
        (
            "test_projections.py::test_golden_log_pure_folds",
            "test_projections.py::test_title_latest_wins_and_empty_title_is_kept_as_state",
            "test_projections.py::test_session_stats_turns_count_distinct_turns",
            "test_projections.py::test_turn_outline_scenario_c_retried_turn_and_steering",
        ),
        "",
    ),
    (
        "PRJ-03",
        "done",
        "persisted cache: identity binding, backup-and-skip, log flushed before cache written",
        (
            "test_projection_cache.py::test_identity_matches_matrix",
            "test_projection_cache.py::test_schema_invalid_record_is_backed_up_and_skipped",
            "integration/test_projection_cache_roundtrip.py::test_kill_at_put_cache_never_ahead_of_log",
        ),
        "",
    ),
    (
        "CHK-01",
        "done",
        "three flush points: before first model chunk, before top-level tool dispatch, before each step",
        (
            "test_checkpoint_policy.py::test_llm_stream_flushes_before_first_chunk",
            "test_checkpoint_policy.py::test_tool_execute_flushes_before_top_level_body",
            "test_checkpoint_policy.py::test_pre_step_flushes_before_step",
        ),
        "",
    ),
    (
        "CHK-02",
        "done",
        "flush rejection blocks dispatch: adapter never invoked, tool body never run, step never entered",
        (
            "test_checkpoint_policy.py::test_llm_stream_failing_flush_never_dispatches",
            "test_checkpoint_policy.py::test_tool_execute_failing_flush_never_runs_body",
            "test_checkpoint_policy.py::test_pre_step_failing_flush_never_enters_step",
            "test_phase2_acceptance.py::test_acceptance_failing_flush_blocks_llm_and_tool_dispatch",
        ),
        "",
    ),
    (
        "CHK-03",
        "done",
        "turn/end performs no flush",
        ("test_checkpoint_policy.py::test_turn_end_performs_no_flush",),
        "",
    ),
    (
        "TTL-01",
        "done",
        "log-backed titles: rename pins, refresh unpins, title survives restart",
        (
            "test_title.py::test_rename_pins_against_automatic_titling",
            "test_title.py::test_refresh_unpins_user_title_without_provider",
            "integration/test_title_roundtrip.py::test_title_survives_restart_and_rename_pins",
            "test_phase2_acceptance.py::test_acceptance_rename_pins_against_auto_title",
        ),
        "",
    ),
    (
        "TTL-02",
        "done",
        "title config required with no library defaults; shipped values 5/40/80",
        (
            "test_title.py::test_resolve_title_config_valid",
            "test_title.py::test_resolve_title_config_missing",
            "test_title.py::test_title_service_direct_construction_requires_config",
            "test_title.py::test_service_config_defaults_fallback_words",
        ),
        "",
    ),
    (
        "TTL-03",
        "done",
        "first-prompt provider contract: first message only, request event logged, invalid results rejected",
        (
            "test_title_first_prompt.py::test_generate_sends_only_first_message",
            "test_title_first_prompt.py::test_generate_logs_request_event_before_dispatch",
            "test_title_first_prompt.py::test_title_from_response_rejects",
        ),
        "",
    ),
    (
        "REQ-01",
        "done",
        "request/header canonical fold with reason taxonomy",
        (
            "test_request_header.py::test_fold_request_header_latest_wins_and_skips_non_headers",
            "test_request_header.py::test_request_header_reasons_are_the_four_taxonomy_values",
            "test_request_header.py::test_canonical_header_drops_empty_tools_and_keeps_nonempty",
        ),
        "",
    ),
    (
        "REQ-02",
        "done",
        "tool-history projection for request reconstruction",
        (
            "test_tool_history.py::test_developer_addition_resolves_from_named_header",
            "test_tool_history.py::test_developer_addition_missing_definition_throws",
            "test_tool_history.py::test_first_header_starts_baseline_series",
        ),
        "",
    ),
    (
        "REQ-03",
        "done",
        "assistant attempt vs message; developer and system preservation",
        (
            "test_format.py::test_v1_to_v2_failed_attempt_emitted_at_close",
            "test_store.py::test_append_assistant_settlement_admission",
            "test_store.py::test_append_developer_plugin_source_refused",
            "test_phase2_acceptance.py::test_acceptance_replay_recorded_turn_reproduces_identical_sequence",
        ),
        "",
    ),
    (
        "REQ-04",
        "done",
        "message projection rewrite: no adjacent same-role merging, no synthesized compaction summary",
        (
            "test_store.py::test_derive_messages_never_merges_adjacent_roles",
            "test_store.py::test_derive_messages_compaction_summary_is_log_only",
            "test_phase2_acceptance.py::test_acceptance_compaction_summary_is_real_message_not_synthesized",
        ),
        "",
    ),
    (
        "RN-01",
        "done",
        "workspace-snapshot rename in the plugins repo; no harness code changed",
        (),
        "Proved in the plugins repo: test_snapshot_capture_diff_and_restore; no harness test applies.",
    ),
    (
        "ACC-01",
        "done",
        "round-trip against v0-v3 fixtures and recorded traces (CI half)",
        (
            "test_phase2_acceptance.py::test_acceptance_pre_step_v0_rows_refused_on_load",
            "test_phase2_acceptance.py::test_acceptance_pre_step_v0_rows_refused_by_persistence_open",
            "test_phase2_acceptance.py::test_acceptance_valid_v2_chronology_migrates_and_replays",
            "test_phase2_acceptance.py::test_acceptance_harness_written_session_reloads_with_identical_messages",
            "test_phase2_acceptance.py::test_acceptance_v2_and_v4_generations_replay_identical_sequence",
        ),
        "",
    ),
    (
        "ACC-02",
        "done",
        "replay of a recorded session reproduces the identical message sequence",
        (
            "test_phase2_acceptance.py::test_acceptance_replay_recorded_turn_reproduces_identical_sequence",
            "test_phase2_acceptance.py::test_acceptance_compaction_summary_is_real_message_not_synthesized",
        ),
        "",
    ),
    (
        "ACC-03",
        "done",
        "rename pins against auto-title",
        ("test_phase2_acceptance.py::test_acceptance_rename_pins_against_auto_title",),
        "",
    ),
    (
        "ACC-04",
        "done",
        "checkpoint rejection prevents adapter dispatch",
        (
            "test_phase2_acceptance.py::test_acceptance_failing_flush_blocks_llm_and_tool_dispatch",
        ),
        "",
    ),
    (
        "ACC-05",
        "done",
        "corrupt / unknown-version files are refused, not silently repaired",
        (
            "test_phase2_acceptance.py::test_acceptance_corrupt_row_refused_not_repaired",
            "test_phase2_acceptance.py::test_acceptance_unknown_version_refused_without_decoding_rows",
            "test_phase2_acceptance.py::test_acceptance_unknown_non_ignorable_type_refused_at_load",
        ),
        "",
    ),
    (
        "ACC-06",
        "not-done",
        "live cross-tooling round-trip: a session written here opens in the other tooling and vice versa",
        (),
        "Manual gate, not a CI gate: it needs a built install of the other tooling, and no built install exists in CI. Phase 2 is not declared done until it has been run once.",
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
    (
        "DEF-01",
        "deferred",
        "compression: plain JSONL in Phase 2; a compressed generation is a hard error",
        (
            "test_persistence_jsonl.py::test_resolve_current_generation_zstd_is_hard_error",
        ),
        "Decided deferral: revisit only with evidence of size pressure.",
    ),
    (
        "DEF-02",
        "deferred",
        "repair command: refuse-by-default, no repair tool in Phase 2",
        ("test_store.py::test_session_load_refuses_torn_row",),
        "Decided: refusal is the behavior; a repair command, if wanted, is a later decision and must never be the silent default again.",
    ),
    (
        "DEF-03",
        "deferred",
        "non-streaming model call path carries no durability checkpoint",
        (),
        "Follow-up for Phase 3 loop work: route the non-streaming turn through the streaming path or checkpoint the call the same way.",
    ),
    (
        "DEF-04",
        "not-done",
        "crash-recovery end-to-end: kill mid-write, log prefix survives",
        (),
        "Not built here: needs a built install and subprocess choreography. Deferred to Phase 3 if wanted.",
    ),
    (
        "OOS-01",
        "out-of-scope",
        "session query and search UI",
        (),
        "Explicitly out of Phase 2 scope; consumes the core built here.",
    ),
    (
        "OOS-02",
        "out-of-scope",
        "telemetry",
        (),
        "Explicitly out of Phase 2 scope (Phase 6/7 or out).",
    ),
    (
        "OOS-03",
        "out-of-scope",
        "message-quotes (UI concern)",
        (),
        "Explicitly out of Phase 2 scope.",
    ),
    (
        "OOS-04",
        "out-of-scope",
        "title providers beyond first-prompt",
        (),
        "Explicitly out of Phase 2 scope; the all-prompts provider is experimental.",
    ),
    (
        "OOS-05",
        "out-of-scope",
        "compaction engines",
        (),
        "Explicitly out of Phase 2 scope (Phase 4). The four compaction event types exist as vocabulary only; no writer is built here.",
    ),
    (
        "OOS-06",
        "out-of-scope",
        "log-upload behavior",
        (),
        "Explicitly out of Phase 2 scope. The acceptance-watermark fold is in (FMT-06); the upload itself is deferred, not deleted.",
    ),
)

# The complete inventory, stated independently of the table above. A row
# may not be added or removed without updating this set in the same
# change, so coverage cannot shrink silently.
EXPECTED_ROW_IDS: frozenset[str] = frozenset(
    {
        "FMT-01",
        "FMT-02",
        "FMT-03",
        "FMT-04",
        "FMT-05",
        "FMT-06",
        "EVT-01",
        "EVT-02",
        "EVT-03",
        "EVT-04",
        "STO-01",
        "STO-02",
        "STO-03",
        "STO-04",
        "PER-01",
        "PER-02",
        "PER-03",
        "PER-04",
        "PRJ-01",
        "PRJ-02",
        "PRJ-03",
        "CHK-01",
        "CHK-02",
        "CHK-03",
        "TTL-01",
        "TTL-02",
        "TTL-03",
        "REQ-01",
        "REQ-02",
        "REQ-03",
        "REQ-04",
        "RN-01",
        "ACC-01",
        "ACC-02",
        "ACC-03",
        "ACC-04",
        "ACC-05",
        "ACC-06",
        "PIN-01",
        "PIN-02",
        "DEF-01",
        "DEF-02",
        "DEF-03",
        "DEF-04",
        "OOS-01",
        "OOS-02",
        "OOS-03",
        "OOS-04",
        "OOS-05",
        "OOS-06",
    }
)

_TESTS_ROOT = Path(__file__).parents[3]
_SESSION_DIR = _TESTS_ROOT / "unit" / "magitech_harness" / "session"


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
    """Split a node id into its file and bare test name."""
    relative, name = node_id.split("::")
    name = name.split("[", 1)[0]
    if relative.startswith("integration/"):
        return _TESTS_ROOT / relative, name
    return _SESSION_DIR / relative, name


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


def test_conformance_live_round_trip_stays_not_done() -> None:
    """The live cross-tooling round-trip is a manual gate, never a CI claim.

    Guards against the row being silently flipped to done: until the
    manual run happens, the table must keep saying so.
    """
    row = next(row for row in CONFORMANCE_ROWS if row[0] == "ACC-06")
    assert row[1] == "not-done"
    assert row[3] == ()
    assert "Manual gate" in row[4]


def test_conformance_pinned_divergences_stay_pinned() -> None:
    """The two pinned divergences stay pinned, not fixed and not dropped."""
    pinned = {row[0]: row for row in CONFORMANCE_ROWS if row[1] == "pinned"}
    assert set(pinned) == {"PIN-01", "PIN-02"}
    for row in pinned.values():
        assert "design decision" in row[4]
