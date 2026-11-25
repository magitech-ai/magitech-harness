"""Phase 8 C4 conformance suite: findings, fixtures, and ACC-06.

One row per acceptance item. A row is ``done`` only when every node it cites
exists and passes; node ids are resolved by static inspection, so the map cannot
go stale without this suite failing. Harness node ids are resolved under
``tests/``; plugins-owned rows name their cross-repo proof in the note, matching
the phase 2 conformance convention (harness CI does not check out mth-components).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_HARNESS = Path(__file__).resolve().parents[3]
_TESTS = _HARNESS / "tests"
_UNIT = _TESTS / "unit" / "magitech_harness"

# (id, status, item, harness_node_ids, note)
C4_ROWS: tuple[tuple[str, str, str, tuple[str, ...], str], ...] = (
    # --- the 20 retrospective findings, mapped to their live probes ---
    (
        "FIND-01",
        "done",
        "one unified agent driver (legacy retired)",
        (
            "integration/test_c4_single_driver_surface.py::test_single_agent_driver_row_and_no_legacy_module",
        ),
        "",
    ),
    (
        "FIND-02",
        "done",
        "pre-step reject ends the turn with no step",
        (
            "integration/test_harness_integration.py::test_waterfall_guard_pre_step_veto",
        ),
        "",
    ),
    (
        "FIND-03",
        "done",
        "non-streaming model call is cancellable",
        (
            "unit/magitech_harness/session/test_title_first_prompt.py::test_generate_aborted_signal_raises_before_dispatch",
        ),
        "",
    ),
    (
        "FIND-04",
        "done",
        "turn-end taxonomy is structured and closed",
        (
            "unit/magitech_harness/agent/test_react_loop.py::test_react_loop_pre_step_reject_and_no_adapter_middleware",
        ),
        "",
    ),
    (
        "FIND-05",
        "done",
        "one tool service surface and one executor",
        (
            "integration/test_c4_single_driver_surface.py::test_single_tool_surface_and_one_executor",
        ),
        "",
    ),
    (
        "FIND-06",
        "done",
        "token meter derives legacy-shaped usage without KeyError",
        (
            "unit/magitech_harness/llm/test_token_meter.py::test_derive_turn_token_usage",
        ),
        "",
    ),
    (
        "FIND-07",
        "done",
        "retry live on the unified driver; legacy loop absent",
        (
            "integration/test_phase3_acceptance.py::test_phase3_retry_rate_limit_five_retries_and_model_listing_cache",
        ),
        "retired half proved by FIND-01 structural probe",
    ),
    (
        "FIND-08",
        "done",
        "CodeAct escape battery fails closed",
        (),
        "mth-components: tests/unit/mth_components/tools/codeact/test_sandbox_escape.py::test_escape_code_rejected",
    ),
    (
        "FIND-09",
        "done",
        "default approval guard mounted and fails closed",
        (
            "integration/test_approval_guard.py::test_no_approver_mutating_tool_fails_closed_with_audit",
        ),
        "",
    ),
    (
        "FIND-10",
        "done",
        "pipeline approval seam resolves ask decisions",
        (
            "integration/test_phase3_acceptance.py::test_phase3_approval_fail_closed_in_pipeline",
        ),
        "",
    ),
    (
        "FIND-11",
        "done",
        "one-shot grants are bound and never blanket-persisted",
        (),
        "mth-components: tests/unit/mth_components/guards/approval/test_guard.py::test_project_grant_not_persisted_by_default",
    ),
    (
        "FIND-12",
        "done",
        "guards accept both pipeline and legacy payload shapes",
        (),
        "mth-components: tests/unit/mth_components/guards/approval/test_guard.py::test_guard_tool_pre_execute_interception",
    ),
    (
        "FIND-13",
        "done",
        "sandbox enforcement defaults on (fail closed)",
        (),
        "mth-components: tests/unit/mth_components/sandbox/test_ladder.py::test_sandbox_enforce_defaults_to_true",
    ),
    (
        "FIND-14",
        "done",
        "approval policy/turn/audit pair appended in order",
        (
            "unit/magitech_harness/approval/test_service.py::test_audit_pair_appended_in_order",
        ),
        "",
    ),
    (
        "FIND-15",
        "done",
        "compaction surface replace, token budgets (0.8/0.16), and wired pruner",
        (),
        "mth-components: tests/unit/mth_components/guards/test_compaction.py::test_compact_session_with_token_spec, tests/unit/mth_components/guards/compaction/test_config.py::test_budget_formulas, tests/unit/mth_components/guards/test_compaction.py::test_component_registers_pruner_and_prunes_session; harness: unit/magitech_harness/compaction/test_service.py::test_compaction_service_registers_engines_and_pruners",
    ),
    (
        "FIND-16",
        "done",
        "shell/truncation output is marked, never silent",
        (),
        "mth-components: tests/unit/mth_components/guards/test_truncation.py::test_truncation_new_pipeline_path_truncates_text_blocks",
    ),
    (
        "FIND-17",
        "done",
        "subagent children run with the mounted bundle seams",
        (),
        "mth-components: tests/integration/test_subagent_bundle_child.py::test_subagent_turn_on_bundle_child_persists",
    ),
    (
        "FIND-18",
        "done",
        "repeat reminder fires at consecutive {3,5,8} only",
        (
            "integration/test_loop_hygiene_guards.py::test_repeat_chain_consecutive_reset",
        ),
        "mth-components: tests/unit/mth_components/guards/repeat_tool_reminder/test_guard.py::test_default_thresholds",
    ),
    (
        "FIND-19",
        "done",
        "TOOL_TIMEOUT structured for sync and async tools",
        (
            "integration/test_loop_hygiene_guards.py::test_timeout_guard_structures_timeout_for_sync_tool",
        ),
        "mth-components: tests/unit/mth_components/guards/timeout_policy/test_policy.py::test_on_execute_timeout_replaces_result",
    ),
    (
        "FIND-20",
        "done",
        "filesystem tools confine paths to the workspace",
        (),
        "mth-components: tests/unit/mth_components/tools/test_path_confinement.py::test_confined_path_rejects_dotdot_and_symlink",
    ),
    # --- recorded v4 fixtures replayed through the harness ---
    (
        "FIX-01",
        "done",
        "recorded basic-turn replay derives exact messages",
        (
            "integration/test_c4_fixture_replay.py::test_fixture_replay_basic_turn_and_tool_pair",
        ),
        "",
    ),
    (
        "FIX-02",
        "done",
        "recorded title fixture replays the pinned title",
        (
            "integration/test_c4_fixture_replay.py::test_fixture_replay_title_projection",
        ),
        "",
    ),
    (
        "FIX-03",
        "done",
        "recorded compaction checkpoint replaces shadowed span",
        (
            "integration/test_c4_fixture_replay.py::test_fixture_replay_compaction_checkpoint",
        ),
        "",
    ),
    (
        "FIX-04",
        "done",
        "recorded approval audit pairs replay log-only",
        (
            "integration/test_c4_fixture_replay.py::test_fixture_replay_approval_audit_pairs",
        ),
        "",
    ),
    # --- carries the round-trip gate from Phase 2 ---
    (
        "ACC-06",
        "done",
        "bidirectional v4 wire and recorded-fixture round-trip",
        (
            "integration/test_cross_tooling_roundtrip.py::test_bidirectional_v4_session_roundtrip",
            "integration/test_cross_tooling_roundtrip.py::test_bidirectional_v4_catalog_team_and_projection_roundtrip",
            "integration/test_c4_fixture_replay.py::test_fixture_replay_compaction_checkpoint",
        ),
        "see docs/conformance/acc-06.md for the live run record",
    ),
)

EXPECTED_ROW_IDS = frozenset(row[0] for row in C4_ROWS)


def _defined_test_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name.startswith("test_")
    }


def _resolve(node_id: str) -> tuple[Path, str]:
    relative, name = node_id.split("::")
    return (_TESTS / relative), name.split("[", 1)[0]


_LISTED = [(row[0], node) for row in C4_ROWS for node in row[3]]


@pytest.mark.parametrize(
    ("row_id", "node_id"), _LISTED, ids=[f"{r}:{n}" for r, n in _LISTED]
)
def test_c4_listed_test_exists(row_id: str, node_id: str) -> None:
    path, name = _resolve(node_id)
    assert path.is_file(), f"{row_id}: {node_id} names a missing file"
    assert name in _defined_test_names(path), (
        f"{row_id}: {node_id} names a missing test"
    )


def test_c4_all_rows_done() -> None:
    not_done = [row[0] for row in C4_ROWS if row[1] != "done"]
    assert not_done == [], f"non-done C4 rows: {not_done}"


def test_c4_done_rows_have_proof() -> None:
    unproven = [
        row[0]
        for row in C4_ROWS
        if row[1] == "done" and not row[3] and "mth-components:" not in row[4]
    ]
    assert unproven == [], f"done rows without a live probe: {unproven}"


def test_c4_row_ids_unique_and_complete() -> None:
    ids = [row[0] for row in C4_ROWS]
    assert len(ids) == len(set(ids)), "duplicate C4 row ids"
    assert len(C4_ROWS) == 25, "20 findings + 4 fixtures + ACC-06"
    assert frozenset(ids) == EXPECTED_ROW_IDS


def test_c4_findings_cover_all_twenty() -> None:
    findings = [row[0] for row in C4_ROWS if row[0].startswith("FIND-")]
    assert findings == [f"FIND-{i:02d}" for i in range(1, 21)]
