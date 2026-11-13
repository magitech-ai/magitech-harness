"""Unit tests for the durable session format (v4, migrations, catalog).

Migration fixtures are modeled on the reference format's own spec shapes;
only minimal fixture data is included here. The end-to-end fixtures assert
byte-equivalent v4 output under canonical JSON serialization.
"""

import hashlib
import json
from typing import Any

import pytest

from magitech_harness.session.format import (
    HISTORICAL_SESSION_FORMAT_CATALOG,
    KNOWN_SESSION_EVENT_TYPES,
    RELEASED_V3_EVENT_TYPES,
    SESSION_FORMAT_CATALOG,
    SESSION_FORMAT_V0_TO_V1,
    SESSION_FORMAT_V1_TO_V2,
    SESSION_FORMAT_V2_TO_V3,
    SESSION_FORMAT_V3_TO_V4,
    AssistantStreamAccumulator,
    PackedAssistantChunkRun,
    SessionFormatArtifact,
    SessionFormatCatalog,
    SessionFormatChain,
    SessionFormatError,
    SessionFormatEventRun,
    SessionFormatMigration,
    SessionFormatMigrationStageInput,
    SessionFormatUnsupportedMigrationError,
    _append_stream_record,
    _assert_content_block,
    _assert_content_kind,
    _assert_content_kinds,
    _assert_event_v2,
    _assert_header_version,
    _assert_message_source_v2,
    _assert_owned_content,
    _assert_tool_change,
    _AttemptGroup,
    _canonicalize_transformed_event,
    _ChainedContext,
    _Collector,
    _Decoder,
    _Emit,
    _finish_decoder,
    _is_repair_identity,
    _LegacyCompactionState,
    _map_event_messages,
    _map_one,
    _matches_chunk_sources,
    _migrate_v3_block,
    _migrate_v3_chunk,
    _migrate_v3_content,
    _MigratingSessionFormatRestore,
    _normalize_legacy_compaction,
    _normalize_legacy_error_reason,
    _normalize_legacy_message,
    _normalize_legacy_request_header,
    _normalize_legacy_retry,
    _normalize_legacy_steering,
    _normalize_legacy_turn_end,
    _producer_kind,
    _refusal,
    _remap_event_v2_to_v3,
    _remap_payload_references,
    _remap_v3_references,
    _rename_ptc_event,
    _replacement_start,
    _restore_current_v4,
    _restore_current_version,
    _restore_released_v2_artifact,
    _result_content,
    _rewrite_event_message_source,
    _rewrite_plugin_source,
    _Stage,
    _V01Codec,
    _V2Codec,
    _V3Codec,
    _V4Codec,
    accepted_through,
    assert_released_event_payload,
    assert_released_surface_metadata,
    assert_released_v4_relationships,
    assert_v0_v1_header,
    assert_v1_header,
    assert_v3_event,
    assert_v3_event_admission,
    assert_v3_header,
    assert_v3_structural_row,
    assert_v4_assistant_settlement_data,
    assert_v4_developer_data,
    assert_v4_fork_result,
    assert_v4_header,
    assert_v4_message_sources,
    assert_v4_request_context_data,
    assert_v4_request_header_data,
    assert_v4_retired_syntax,
    assert_v4_row_admission,
    assert_v4_source_row_admission,
    assert_v4_system_message_fields,
    assert_v4_tool_result_message,
    catalog_fact,
    child_catalog_fact,
    child_catalog_source,
    child_catalog_subject,
    create_session_format_catalog,
    create_session_format_catalog_with_children,
    create_session_format_v3_to_v4,
    decode_envelope_row,
    decode_seq_ranges,
    define_session_format_migration,
    encode_envelope_event,
    encode_seq_ranges,
    exact_keys,
    historical_child_catalog_source,
    inspect_format_version,
    lift_tool_result,
    migrate_v3_event_content,
    parse_session_format_log_filename,
    require_safe_integer,
    restore_released_v3_artifact,
    restore_released_v4_artifact,
    rewrite_v3_message_source,
    session_format_log_filename,
    session_format_version_refusal,
    snapshot_header,
    validate_delivery_accepted,
)


def _dump(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _header(version: int, session_id: str = "s", **extra: object) -> dict:
    header: dict = {
        "type": "session",
        "version": version,
        "id": session_id,
        "createdAt": 1000,
        "delegationDepth": 0,
    }
    if version >= 2:
        header["isSeeded"] = False
    header.update(extra)
    return header


def _migrate(
    header: dict, rows: list[dict], children: list | None = None
) -> SessionFormatArtifact:
    """Restore one stored artifact through the shipped chain into v4."""
    catalog = create_session_format_catalog_with_children(
        [] if children is None else children
    )
    restore = catalog.create_restore(
        dict(header), validation="transformed", recovery="strict"
    )
    for row in rows:
        restore.decode_row(dict(row))
    return restore.finish()


def _encode_lines(artifact: SessionFormatArtifact) -> list[str]:
    lines = [
        _dump(
            SESSION_FORMAT_CATALOG.encode_current_header(
                artifact.header, artifact.inherited_event_count
            )
        )
    ]
    for event in artifact.events:
        lines.append(_dump(SESSION_FORMAT_CATALOG.encode_current_event(event)))
    return lines


def _system_id(session_id: str, seq: int, event_type: str) -> str:
    identity = json.dumps(
        ["session-format-v2-to-v3", session_id, seq, event_type],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "v2-to-v3-system-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _run_stage(
    migration, header: dict, cut: int | None, rows: list[dict], kind: str = "decoded"
) -> tuple[dict, int, list[dict]]:
    """Drive one adjacent migration stage in isolation."""
    from magitech_harness.session.format import SessionFormatMigrationStageInput

    logical = {
        key: value for key, value in header.items() if key not in ("type", "seedLength")
    }
    # The logical header carries isSeeded, derived from seedLength for
    # v0/v1 exactly as the released codec derives it on decode.
    if "isSeeded" not in logical:
        logical["isSeeded"] = "seedLength" in header or header.get("isSeeded") is True
    target_header = migration.migrate_header(dict(logical))
    stage = migration.create_stage(
        SessionFormatMigrationStageInput(
            source_header=dict(logical),
            target_header=target_header,
            source_inherited_event_count=cut,
            source_kind=kind,
        )
    )
    out = _Collector()
    for row in rows:
        stage.transform_event(dict(row), out)
    return target_header, stage.finish(out), out.values


# ---------------------------------------------------------------------------
# Header classification
# ---------------------------------------------------------------------------


def test_read_header_current() -> None:
    result = SESSION_FORMAT_CATALOG.read_header(_header(4, "cur"))
    assert result.status == "current"
    assert result.stored_version == 4
    assert result.target_version == 4
    assert result.header is not None
    assert result.header["version"] == 4
    assert result.header["id"] == "cur"


def test_read_header_migration_required() -> None:
    for version in (0, 1, 2, 3):
        result = SESSION_FORMAT_CATALOG.read_header(_header(version, f"v{version}"))
        assert result.status == "migration-required"
        assert result.stored_version == version
        assert result.header is not None
        assert result.header["version"] == 4


def test_read_header_newer_version_is_unsupported() -> None:
    result = SESSION_FORMAT_CATALOG.read_header(_header(9, "new"))
    assert result.status == "unsupported"
    assert result.stored_version == 9
    assert result.reason == "stored Session uses newer format v9; this build writes v4"


def test_read_header_malformed() -> None:
    assert SESSION_FORMAT_CATALOG.read_header({"type": "session"}).status == "malformed"
    assert SESSION_FORMAT_CATALOG.read_header("not an object").status == "malformed"
    assert SESSION_FORMAT_CATALOG.read_header(None).status == "malformed"
    assert (
        SESSION_FORMAT_CATALOG.read_header({**_header(4, "bad"), "version": "4"}).status
        == "malformed"
    )
    result = SESSION_FORMAT_CATALOG.read_header(_header(4, "bad", isSeeded="yes"))
    assert result.status == "malformed"


def test_read_header_migrates_agent_preset() -> None:
    result = SESSION_FORMAT_CATALOG.read_header(_header(2, "x", agentPreset="code"))
    assert result.status == "migration-required"
    assert result.header is not None
    assert result.header["agentPreset"] == "ptc"


def test_read_header_malformed_seed_length() -> None:
    header = _header(1, "x", seedLength="not-a-count")
    assert SESSION_FORMAT_CATALOG.read_header(header).status == "malformed"


def test_session_format_version_refusal_text() -> None:
    assert session_format_version_refusal("abc", 9) == (
        'session "abc" uses log format v9, but this harness reads only v4: '
        "the log was written by a newer harness — upgrade the harness to open it"
    )
    assert session_format_version_refusal("abc", 2) == (
        'session "abc" uses log format v2, older than the supported v4, '
        "and this build ships no upgrade path for it"
    )


def test_inspect_format_version() -> None:
    assert inspect_format_version({"version": 3}) == 3
    with pytest.raises(SessionFormatError):
        inspect_format_version({"version": -1})
    with pytest.raises(SessionFormatError):
        inspect_format_version({"version": 1.5})
    with pytest.raises(SessionFormatError):
        inspect_format_version([3])


def test_historical_catalog_reads_v3_as_current() -> None:
    result = HISTORICAL_SESSION_FORMAT_CATALOG.read_header(_header(3, "hist"))
    assert result.status == "current"
    assert result.header is not None
    assert result.header["version"] == 3
    assert (
        HISTORICAL_SESSION_FORMAT_CATALOG.read_header(_header(4)).status
        == "unsupported"
    )


# ---------------------------------------------------------------------------
# Catalog construction rules
# ---------------------------------------------------------------------------


def _catalog_kwargs(**overrides):  # type: ignore[no-untyped-def]
    kwargs = {
        "current_version": 4,
        "codecs": [_V01Codec(0), _V01Codec(1), _V2Codec(), _V3Codec(), _V4Codec()],
        "current_encoder": _V4Codec(),
        "migrations": [
            SESSION_FORMAT_V0_TO_V1,
            SESSION_FORMAT_V1_TO_V2,
            SESSION_FORMAT_V2_TO_V3,
            SESSION_FORMAT_V3_TO_V4,
        ],
        "restore_current": lambda artifact: artifact,
        "restore_transformed_current": lambda artifact: artifact,
        "restore_current_header": assert_v4_header,
    }
    kwargs.update(overrides)
    return kwargs


def test_catalog_duplicate_codec_throws() -> None:
    with pytest.raises(SessionFormatError, match="codec v0 is duplicated"):
        create_session_format_catalog(
            **_catalog_kwargs(
                codecs=[
                    _V01Codec(0),
                    _V01Codec(0),
                    _V01Codec(1),
                    _V2Codec(),
                    _V3Codec(),
                    _V4Codec(),
                ]
            )
        )


def test_catalog_missing_codec_throws() -> None:
    with pytest.raises(SessionFormatError, match="codec v3 is missing"):
        create_session_format_catalog(
            **_catalog_kwargs(
                codecs=[_V01Codec(0), _V01Codec(1), _V2Codec(), _V4Codec()]
            )
        )


def test_catalog_newer_codec_throws() -> None:
    class _FutureCodec:
        version = 5

        def decode_header(self, value):  # type: ignore[no-untyped-def]
            raise AssertionError("must not decode")

        def create_decoder(self, value, recovery):  # type: ignore[no-untyped-def]
            raise AssertionError("must not decode")

    with pytest.raises(SessionFormatError, match="codec v5 is newer than current v4"):
        create_session_format_catalog(
            **_catalog_kwargs(
                codecs=[
                    _V01Codec(0),
                    _V01Codec(1),
                    _V2Codec(),
                    _V3Codec(),
                    _V4Codec(),
                    _FutureCodec(),
                ]
            )
        )


def test_catalog_missing_migration_throws() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="migration v2->v3 is missing"
    ):
        create_session_format_catalog(
            **_catalog_kwargs(
                migrations=[
                    SESSION_FORMAT_V0_TO_V1,
                    SESSION_FORMAT_V1_TO_V2,
                    SESSION_FORMAT_V3_TO_V4,
                ]
            )
        )


def test_catalog_duplicate_migration_throws() -> None:
    with pytest.raises(SessionFormatError, match="is duplicated"):
        create_session_format_catalog(
            **_catalog_kwargs(
                migrations=[
                    SESSION_FORMAT_V0_TO_V1,
                    SESSION_FORMAT_V0_TO_V1,
                    SESSION_FORMAT_V2_TO_V3,
                    SESSION_FORMAT_V3_TO_V4,
                ]
            )
        )


def test_encode_current_header_rejects_non_current() -> None:
    with pytest.raises(SessionFormatError, match="encodeCurrent requires"):
        SESSION_FORMAT_CATALOG.encode_current_header(_header(3), 0)


# ---------------------------------------------------------------------------
# Filenames and seq ranges
# ---------------------------------------------------------------------------


def test_session_format_log_filename() -> None:
    assert session_format_log_filename(0) == "session.jsonl"
    assert session_format_log_filename(4) == "session.v4.jsonl"
    assert parse_session_format_log_filename("session.jsonl") == 0
    assert parse_session_format_log_filename("session.v3.jsonl") == 3
    assert parse_session_format_log_filename("session.v04.jsonl") is None
    assert parse_session_format_log_filename("session.V4.jsonl") is None
    assert parse_session_format_log_filename("other.jsonl") is None
    assert parse_session_format_log_filename("session.v9.jsonl") == 9


def test_encode_seq_ranges_round_trip() -> None:
    assert encode_seq_ranges([]) == []
    assert encode_seq_ranges([5]) == [5]
    assert encode_seq_ranges([1, 2]) == [1, 2]
    assert encode_seq_ranges([1, 2, 3]) == [[1, 3]]
    assert encode_seq_ranges([0, 1, 2, 5, 6, 7, 8, 10]) == [[0, 2], [5, 8], 10]
    assert decode_seq_ranges([[0, 2], [5, 8], 10]) == [0, 1, 2, 5, 6, 7, 8, 10]
    assert decode_seq_ranges([3, 1, 2]) == [3, 1, 2]


def test_decode_seq_ranges_rejects_bad_ranges() -> None:
    with pytest.raises(SessionFormatError):
        decode_seq_ranges([[2, 1]])
    with pytest.raises(SessionFormatError):
        decode_seq_ranges([[1, 3], 2])
    with pytest.raises(SessionFormatError):
        decode_seq_ranges([1, "x"])
    with pytest.raises(SessionFormatError):
        decode_seq_ranges([[0, 5]], max_entries=3)


# ---------------------------------------------------------------------------
# Delivery-acceptance watermark fold
# ---------------------------------------------------------------------------


def _delivery(seq: int, through: int, session_id: str = "s", version: int = 4) -> dict:
    return {
        "type": "session-log-deepseek/delivery-accepted",
        "seq": seq,
        "time": 1000 + seq,
        "data": {
            "sessionId": session_id,
            "throughSeq": through,
            "sessionFormatVersion": version,
        },
    }


def test_accepted_through_folds_matching_markers() -> None:
    events = [
        _delivery(2, 1),
        _delivery(5, 4),
        _delivery(7, 3, session_id="other"),
        _delivery(9, 6, version=3),
    ]
    assert accepted_through(events, 4, "s") == 4
    assert accepted_through([], 4, "s") == -1


def test_accepted_through_rejects_malformed_watermark() -> None:
    with pytest.raises(SessionFormatError):
        accepted_through([_delivery(2, 2)], 4, "s")
    with pytest.raises(SessionFormatError):
        accepted_through([_delivery(2, 1, session_id="")], 4, "s")


def test_validate_delivery_accepted() -> None:
    assert validate_delivery_accepted(_delivery(2, 1), 4) == "s"
    assert validate_delivery_accepted(_delivery(2, 1, version=3), 4) is None
    assert (
        validate_delivery_accepted(
            {"type": "turn/start", "seq": 0, "time": 1, "data": {}}, 4
        )
        is None
    )
    with pytest.raises(SessionFormatError):
        validate_delivery_accepted(_delivery(2, 5), 4)


# ---------------------------------------------------------------------------
# Migration edge v0 -> v1
# ---------------------------------------------------------------------------

V0_HEADER = _header(0, "v0")


def test_v0_to_v1_turn_start_trigger_stripped() -> None:
    _, _, events = _run_stage(
        SESSION_FORMAT_V0_TO_V1,
        V0_HEADER,
        0,
        [
            {
                "type": "turn/start",
                "seq": 0,
                "time": 1,
                "data": {"turn": 1, "trigger": {"kind": "user"}},
            }
        ],
    )
    assert events == [{"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}]


def test_v0_to_v1_turn_start_without_trigger_passes() -> None:
    _, _, events = _run_stage(
        SESSION_FORMAT_V0_TO_V1,
        V0_HEADER,
        0,
        [{"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}],
    )
    assert events[0]["data"] == {"turn": 1}


def test_v0_to_v1_turn_start_bad_trigger_is_malformed() -> None:
    with pytest.raises(SessionFormatError, match="malformed pre-react-loop"):
        _run_stage(
            SESSION_FORMAT_V0_TO_V1,
            V0_HEADER,
            0,
            [
                {
                    "type": "turn/start",
                    "seq": 0,
                    "time": 1,
                    "data": {"turn": 0, "trigger": {"kind": "user"}},
                }
            ],
        )


def test_v0_to_v1_turn_end_reason_normalization() -> None:
    _, _, events = _run_stage(
        SESSION_FORMAT_V0_TO_V1,
        V0_HEADER,
        0,
        [
            {
                "type": "turn/end",
                "seq": 0,
                "time": 1,
                "data": {"turn": 1, "reason": {"kind": "disposed"}},
            },
            {
                "type": "turn/end",
                "seq": 1,
                "time": 2,
                "data": {"turn": 2, "reason": {"kind": "aborted"}},
            },
            {
                "type": "turn/end",
                "seq": 2,
                "time": 3,
                "data": {"turn": 3, "reason": {"kind": "completed"}},
            },
            {
                "type": "turn/end",
                "seq": 3,
                "time": 4,
                "data": {
                    "turn": 4,
                    "reason": {"kind": "error", "step": 1, "message": "boom"},
                },
            },
        ],
    )
    assert events[0]["data"]["reason"] == {
        "kind": "aborted",
        "reason": {"kind": "disposed"},
    }
    assert events[1]["data"]["reason"] == {
        "kind": "aborted",
        "reason": {"kind": "legacy"},
    }
    assert events[2]["data"]["reason"] == {"kind": "completed"}
    assert events[3]["data"]["reason"] == {
        "kind": "error",
        "error": {"message": "boom", "code": "UNKNOWN"},
    }


def test_v0_to_v1_compact_types_renamed() -> None:
    _, _, events = _run_stage(
        SESSION_FORMAT_V0_TO_V1,
        V0_HEADER,
        0,
        [
            {
                "type": "compact/start",
                "seq": 0,
                "time": 1,
                "data": {"compactionId": "c1", "turn": 1},
            },
            {
                "type": "compact/end",
                "seq": 1,
                "time": 2,
                "data": {"compactionId": "c1", "turn": 1},
            },
        ],
    )
    assert [event["type"] for event in events] == ["compaction/start", "compaction/end"]


def test_v0_to_v1_steering_flat_form_wrapped() -> None:
    _, _, events = _run_stage(
        SESSION_FORMAT_V0_TO_V1,
        V0_HEADER,
        0,
        [
            {
                "type": "steering/message",
                "seq": 0,
                "time": 1,
                "data": {
                    "turn": 1,
                    "content": [{"type": "text", "text": "steer"}],
                    "source": {"kind": "user"},
                },
            }
        ],
    )
    assert events == [
        {
            "type": "user/message",
            "seq": 0,
            "time": 1,
            "data": {
                "content": [{"type": "text", "text": "steer"}],
                "source": {"kind": "user"},
                "id": "legacy-message:v0:0",
                "role": "user",
            },
        }
    ]


def test_v0_to_v1_unsupported_legacy_types_refused() -> None:
    for event in (
        {"type": "request/header-delta", "seq": 0, "time": 1, "data": {}},
        {"type": "mode/set", "seq": 0, "time": 1, "data": {}},
        {
            "type": "request/header",
            "seq": 0,
            "time": 1,
            "data": {"header": {}, "reason": "fallback"},
        },
    ):
        with pytest.raises(SessionFormatUnsupportedMigrationError):
            _run_stage(SESSION_FORMAT_V0_TO_V1, V0_HEADER, 0, [event])


def test_v0_to_v1_request_header_message_prefix_dropped() -> None:
    _, _, events = _run_stage(
        SESSION_FORMAT_V0_TO_V1,
        V0_HEADER,
        0,
        [
            {
                "type": "request/header",
                "seq": 0,
                "time": 1,
                "data": {
                    "header": {
                        "messagePrefix": [{"type": "text", "text": "x"}],
                        "tools": [],
                    },
                    "reason": "start",
                },
            }
        ],
    )
    assert events[0]["data"]["header"] == {"tools": []}


def test_v0_to_v1_retry_id_synthesized_and_invalid_refused() -> None:
    _, _, events = _run_stage(
        SESSION_FORMAT_V0_TO_V1,
        V0_HEADER,
        0,
        [
            {
                "type": "llm/retry",
                "seq": 0,
                "time": 1,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "provider": "p",
                    "mode": "m",
                    "policyKey": "k",
                    "retry": 1,
                    "delayMs": 10,
                    "failure": {"message": "x"},
                },
            }
        ],
    )
    assert events[0]["data"]["retryId"] == "legacy-retry:v0:0"
    with pytest.raises(SessionFormatError, match="retryId must be a non-empty string"):
        _run_stage(
            SESSION_FORMAT_V0_TO_V1,
            V0_HEADER,
            0,
            [
                {
                    "type": "llm/retry",
                    "seq": 0,
                    "time": 1,
                    "data": {
                        "retryId": None,
                        "turn": 1,
                        "step": 1,
                        "provider": "p",
                        "mode": "m",
                        "policyKey": "k",
                        "retry": 1,
                        "delayMs": 10,
                        "failure": {"message": "x"},
                    },
                }
            ],
        )


def test_v0_to_v1_compaction_ids_synthesized_and_propagated() -> None:
    _, _, events = _run_stage(
        SESSION_FORMAT_V0_TO_V1,
        V0_HEADER,
        0,
        [
            {"type": "compaction/start", "seq": 0, "time": 1, "data": {"turn": 1}},
            {
                "type": "compaction/summary",
                "seq": 1,
                "time": 2,
                "data": {
                    "summary": [{"type": "text", "text": "s"}],
                    "shadowedRange": {"start": 0, "end": 0},
                    "shadowedSeqs": [0],
                    "shadowedTokenCount": 1,
                    "provider": "p",
                    "model": "m",
                },
            },
            {"type": "compaction/end", "seq": 2, "time": 3, "data": {"turn": 1}},
        ],
    )
    assert events[0]["data"]["compactionId"] == "legacy-compaction:v0:0"
    assert events[1]["data"]["compactionId"] == "legacy-compaction:v0:0"
    assert events[2]["data"]["compactionId"] == "legacy-compaction:v0:0"


def test_v0_to_v1_end_seed_clears_compaction_state() -> None:
    # The end-seed marker clears the open compaction id, so a later
    # summary without its own id is malformed: nothing can supply it.
    with pytest.raises(
        SessionFormatError, match='lacks required member "compactionId"'
    ):
        _run_stage(
            SESSION_FORMAT_V0_TO_V1,
            V0_HEADER,
            0,
            [
                {"type": "compaction/start", "seq": 0, "time": 1, "data": {"turn": 1}},
                {"type": "session/end-seed", "seq": 1, "time": 2, "data": {}},
                {
                    "type": "compaction/summary",
                    "seq": 2,
                    "time": 3,
                    "data": {
                        "summary": [{"type": "text", "text": "s"}],
                        "shadowedRange": {"start": 0, "end": 0},
                        "shadowedSeqs": [0],
                        "shadowedTokenCount": 1,
                        "provider": "p",
                        "model": "m",
                    },
                },
            ],
        )


def test_v0_to_v1_delivery_marker_wrong_session_refused() -> None:
    with pytest.raises(SessionFormatError, match="wrong Session"):
        _run_stage(
            SESSION_FORMAT_V0_TO_V1,
            V0_HEADER,
            0,
            [
                {
                    "type": "session-log-deepseek/delivery-accepted",
                    "seq": 0,
                    "time": 1,
                    "data": {"sessionId": "someone-else", "throughSeq": 0},
                }
            ],
        )


def test_v0_to_v1_packed_run_passes_through() -> None:
    class _RecordingEmit(_Emit):
        def __init__(self) -> None:
            self.runs: list = []
            self.events: list = []

        def emit_event(self, event):  # type: ignore[no-untyped-def]
            self.events.append(event)

        def emit_run(self, run):  # type: ignore[no-untyped-def]
            self.runs.append(run)

    target_header = SESSION_FORMAT_V0_TO_V1.migrate_header(dict(V0_HEADER))
    stage = SESSION_FORMAT_V0_TO_V1.create_stage(
        SessionFormatMigrationStageInput(
            source_header=dict(V0_HEADER),
            target_header=target_header,
            source_inherited_event_count=0,
            source_kind="decoded",
        )
    )
    out = _RecordingEmit()
    run = PackedAssistantChunkRun(
        run_type="released-assistant-chunks",
        first_seq=0,
        event_count=2,
        stream={
            "type": "text-chunks",
            "index": 0,
            "time0": 5,
            "dt": [1],
            "texts": ["a", "b"],
        },
        turn=1,
        step=1,
        last_time=6,
    )
    stage.transform_run(run, out)
    assert out.runs == [run]
    assert out.events == []


def test_v0_to_v1_unknown_type_refused_even_when_ignorable() -> None:
    with pytest.raises(SessionFormatUnsupportedMigrationError):
        _run_stage(
            SESSION_FORMAT_V0_TO_V1,
            V0_HEADER,
            0,
            [
                {
                    "type": "something/new",
                    "seq": 0,
                    "time": 1,
                    "data": {},
                    "ignorable": True,
                }
            ],
        )


# ---------------------------------------------------------------------------
# Migration edge v1 -> v2
# ---------------------------------------------------------------------------

V1_HEADER = _header(1, "v1")
_CHUNK = {"type": "text-delta", "index": 0, "text": "Hello"}


def _v1_stream_rows() -> list[dict]:
    return [
        {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
        {"type": "step/start", "seq": 1, "time": 1001, "data": {"turn": 1, "step": 1}},
        {
            "type": "user/message",
            "seq": 2,
            "time": 1002,
            "data": {
                "role": "user",
                "id": "m1",
                "content": [{"type": "text", "text": "hi"}],
                "source": {"kind": "user"},
            },
            "surfaceOp": "append",
        },
        {
            "type": "assistant/chunk",
            "seq": 3,
            "time": 1003,
            "data": {
                "turn": 1,
                "step": 1,
                "chunk": {"type": "text-delta", "index": 0, "text": "Hello"},
            },
        },
        {
            "type": "assistant/chunk",
            "seq": 4,
            "time": 1004,
            "data": {
                "turn": 1,
                "step": 1,
                "chunk": {"type": "text-delta", "index": 0, "text": " world"},
            },
        },
        {
            "type": "assistant/message",
            "seq": 5,
            "time": 1005,
            "data": {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "m2",
                    "role": "assistant",
                    "source": {"kind": "model"},
                    "content": [{"type": "text", "text": "Hello world"}],
                },
            },
            "sourceEventSeqs": [3, 4],
            "surfaceOp": "append",
        },
        {"type": "step/end", "seq": 6, "time": 1006, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": 7,
            "time": 1007,
            "data": {"turn": 1, "reason": {"kind": "done"}},
        },
    ]


def test_v1_to_v2_stream_embedding() -> None:
    _, cut, events = _run_stage(
        SESSION_FORMAT_V1_TO_V2, V1_HEADER, 0, _v1_stream_rows()
    )
    assert cut == 0
    # A message citing its complete attempt embeds the stream: no
    # separate assistant/attempt event is emitted for a successful run.
    assert [event["type"] for event in events] == [
        "turn/start",
        "step/start",
        "user/message",
        "assistant/message",
        "step/end",
        "turn/end",
    ]
    assert [event["seq"] for event in events] == [0, 1, 2, 3, 4, 5]
    stream = [
        {
            "type": "text-chunks",
            "index": 0,
            "time0": 1003,
            "dt": [1],
            "texts": ["Hello", " world"],
        }
    ]
    assert events[3]["data"]["stream"] == stream
    assert "sourceEventSeqs" not in events[3]
    assert events[3]["surfaceOp"] == "append"


def test_v1_to_v2_failed_attempt_emitted_at_close() -> None:
    _, _, events = _run_stage(
        SESSION_FORMAT_V1_TO_V2,
        V1_HEADER,
        0,
        [
            {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
            {
                "type": "step/start",
                "seq": 1,
                "time": 1001,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "assistant/chunk",
                "seq": 2,
                "time": 1002,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "chunk": {"type": "text-delta", "index": 0, "text": "x"},
                },
            },
            {
                "type": "step/end",
                "seq": 3,
                "time": 1003,
                "data": {"turn": 1, "step": 1},
            },
        ],
    )
    assert [event["type"] for event in events] == [
        "turn/start",
        "step/start",
        "assistant/attempt",
        "step/end",
    ]
    assert events[2]["seq"] == 2
    assert events[2]["data"]["stream"] == [
        {"type": "text-chunks", "index": 0, "time0": 1002, "dt": [], "texts": ["x"]}
    ]


def test_v1_to_v2_goal_split() -> None:
    _, _, events = _run_stage(
        SESSION_FORMAT_V1_TO_V2,
        V1_HEADER,
        0,
        [
            {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
            {
                "type": "user/message",
                "seq": 1,
                "time": 1001,
                "data": {
                    "role": "user",
                    "id": "m1",
                    "content": [{"type": "text", "text": "g"}],
                    "source": {
                        "kind": "goal",
                        "change": {"kind": "goal", "version": 1, "operation": "set"},
                    },
                },
                "surfaceOp": "append",
            },
        ],
    )
    assert events[1] == {
        "type": "goal/change",
        "seq": 1,
        "time": 1001,
        "data": {"kind": "goal", "version": 1, "operation": "set"},
    }
    assert events[2]["type"] == "user/message"
    assert events[2]["data"]["source"] == {"kind": "plugin", "plugin": "goal"}


def test_v1_to_v2_interrupted_turn_synthesized() -> None:
    _, _, events = _run_stage(
        SESSION_FORMAT_V1_TO_V2,
        V1_HEADER,
        0,
        [
            {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
            {
                "type": "agent/inbox/spliced",
                "seq": 1,
                "time": 1001,
                "data": {
                    "target": "next-turn",
                    "start": 0,
                    "inserted": [
                        {
                            "role": "user",
                            "id": "mx",
                            "content": [{"type": "text", "text": "x"}],
                            "source": {"kind": "user"},
                        }
                    ],
                },
            },
            {"type": "turn/start", "seq": 2, "time": 1002, "data": {"turn": 2}},
        ],
    )
    assert [event["type"] for event in events] == [
        "turn/start",
        "agent/inbox/spliced",
        "turn/end",
        "turn/start",
    ]
    assert events[2]["data"] == {"turn": 1, "reason": {"kind": "interrupted"}}


def test_v1_to_v2_turn_start_without_close_refused() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="does not close the prior turn"
    ):
        _run_stage(
            SESSION_FORMAT_V1_TO_V2,
            V1_HEADER,
            0,
            [
                {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
                {"type": "turn/start", "seq": 1, "time": 1001, "data": {"turn": 2}},
            ],
        )


def test_v1_to_v2_unknown_type_refused_even_when_ignorable() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="unknown event type"
    ):
        _run_stage(
            SESSION_FORMAT_V1_TO_V2,
            V1_HEADER,
            0,
            [
                {
                    "type": "something/new",
                    "seq": 0,
                    "time": 1,
                    "data": {},
                    "ignorable": True,
                }
            ],
        )


def test_v1_to_v2_reference_remapping() -> None:
    rows = _v1_stream_rows()
    rows.insert(
        6,
        {
            "type": "session/title",
            "seq": 6,
            "time": 1006,
            "data": {"title": "T", "messageSeqs": [2, 5], "source": {"kind": "user"}},
        },
    )
    rows.insert(
        7,
        {
            "type": "user/message",
            "seq": 7,
            "time": 1007,
            "data": {
                "role": "user",
                "id": "m3",
                "content": [{"type": "text", "text": "again"}],
                "source": {"kind": "user"},
            },
            "surfaceOp": {"op": "replace", "start": 5, "end": 5},
        },
    )
    rows[8]["seq"] = 8
    rows[9]["seq"] = 9
    _, _, events = _run_stage(SESSION_FORMAT_V1_TO_V2, V1_HEADER, 0, rows)
    title = next(event for event in events if event["type"] == "session/title")
    assert title["data"]["messageSeqs"] == [2, 3]
    replaced = next(event for event in events if event["data"].get("id") == "m3")
    assert replaced["surfaceOp"] == {"op": "replace", "start": 3, "end": 3}


def test_v1_to_v2_seeded_marker_tagged_and_cut_recovered() -> None:
    header = _header(1, "seeded", seedLength=2)
    _, cut, events = _run_stage(
        SESSION_FORMAT_V1_TO_V2,
        header,
        2,
        [
            {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
            {
                "type": "step/start",
                "seq": 1,
                "time": 1001,
                "data": {"turn": 1, "step": 1},
            },
            {"type": "session/end-seed", "seq": 2, "time": 1002, "data": {}},
            {
                "type": "user/message",
                "seq": 3,
                "time": 1003,
                "data": {
                    "role": "user",
                    "id": "m1",
                    "content": [{"type": "text", "text": "hi"}],
                    "source": {"kind": "user"},
                },
                "surfaceOp": "append",
            },
        ],
    )
    assert cut == 2
    assert events[2]["data"] == {"inherited": True}


def test_v1_to_v2_seeded_marker_synthesized_when_absent() -> None:
    header = _header(1, "seeded", seedLength=2)
    _, cut, events = _run_stage(
        SESSION_FORMAT_V1_TO_V2,
        header,
        2,
        [
            {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
            {
                "type": "step/start",
                "seq": 1,
                "time": 1001,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "user/message",
                "seq": 2,
                "time": 1002,
                "data": {
                    "role": "user",
                    "id": "m1",
                    "content": [{"type": "text", "text": "hi"}],
                    "source": {"kind": "user"},
                },
                "surfaceOp": "append",
            },
        ],
    )
    assert cut == 2
    assert events[2] == {
        "type": "session/end-seed",
        "seq": 2,
        "time": 1002,
        "data": {"inherited": True},
    }
    assert events[3]["type"] == "user/message"


def test_v1_to_v2_cut_split_attempt_refused() -> None:
    header = _header(1, "seeded", seedLength=2)
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="splits one Assistant attempt"
    ):
        _run_stage(
            SESSION_FORMAT_V1_TO_V2,
            header,
            2,
            [
                {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
                {
                    "type": "assistant/chunk",
                    "seq": 1,
                    "time": 1001,
                    "data": {
                        "turn": 1,
                        "step": 1,
                        "chunk": {"type": "text-delta", "index": 0, "text": "a"},
                    },
                },
                {
                    "type": "assistant/chunk",
                    "seq": 2,
                    "time": 1002,
                    "data": {
                        "turn": 1,
                        "step": 1,
                        "chunk": {"type": "text-delta", "index": 0, "text": "b"},
                    },
                },
            ],
        )


def test_v1_to_v2_decoded_stage_validates_payloads() -> None:
    bad = {
        "type": "user/message",
        "seq": 0,
        "time": 1000,
        "data": {"id": "m1", "content": [], "source": {"kind": "user"}},
    }
    with pytest.raises(SessionFormatError, match="lacks required member"):
        _run_stage(SESSION_FORMAT_V1_TO_V2, V1_HEADER, 0, [bad], kind="decoded")
    _, _, events = _run_stage(
        SESSION_FORMAT_V1_TO_V2, V1_HEADER, 0, [bad], kind="transformed"
    )
    assert events[0]["data"] == bad["data"]


def test_v1_to_v2_message_without_chunk_citation_refused() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="does not cite its complete"
    ):
        _run_stage(
            SESSION_FORMAT_V1_TO_V2,
            V1_HEADER,
            0,
            [
                {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
                {
                    "type": "assistant/chunk",
                    "seq": 1,
                    "time": 1001,
                    "data": {
                        "turn": 1,
                        "step": 1,
                        "chunk": {"type": "text-delta", "index": 0, "text": "a"},
                    },
                },
                {
                    "type": "assistant/message",
                    "seq": 2,
                    "time": 1002,
                    "data": {
                        "turn": 1,
                        "step": 1,
                        "message": {
                            "id": "m2",
                            "role": "assistant",
                            "source": {"kind": "model"},
                            "content": [{"type": "text", "text": "a"}],
                        },
                    },
                    "surfaceOp": "append",
                },
            ],
        )


# ---------------------------------------------------------------------------
# Migration edge v2 -> v3
# ---------------------------------------------------------------------------

V2_HEADER = _header(2, "v2")


def _v2_user(seq: int, time: int, message_id: str, text: str) -> dict:
    return {
        "type": "user/message",
        "seq": seq,
        "time": time,
        "data": {
            "role": "user",
            "id": message_id,
            "content": [{"type": "text", "text": text}],
            "source": {"kind": "user"},
        },
        "surfaceOp": "append",
    }


def _v2_request_header(seq: int, time: int, system: str | None) -> dict:
    header: dict = {} if system is None else {"system": system}
    return {
        "type": "request/header",
        "seq": seq,
        "time": time,
        "data": {"header": header, "reason": {"kind": "start"}},
    }


def _v2_opening() -> list[dict]:
    return [
        {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
        {"type": "step/start", "seq": 1, "time": 1001, "data": {"turn": 1, "step": 1}},
    ]


def _system_source() -> dict:
    return {"kind": "plugin", "plugin": "@deepseek-ai/dsh-system-prompt"}


def test_v2_to_v3_system_promotion() -> None:
    _, cut, events = _run_stage(
        SESSION_FORMAT_V2_TO_V3,
        V2_HEADER,
        0,
        [
            *_v2_opening(),
            _v2_request_header(2, 1002, "Be helpful."),
            _v2_user(3, 1003, "m1", "hi"),
            {
                "type": "step/end",
                "seq": 4,
                "time": 1004,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "turn/end",
                "seq": 5,
                "time": 1005,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
        ],
    )
    assert cut == 0
    assert [event["type"] for event in events] == [
        "turn/start",
        "step/start",
        "system/message",
        "system/message",
        "request/header",
        "user/message",
        "step/end",
        "turn/end",
    ]
    assert [event["seq"] for event in events] == [0, 1, 2, 3, 4, 5, 6, 7]
    head = events[2]
    assert head["time"] == 1001
    assert head["surfaceOp"] == "append"
    assert head["data"] == {
        "turn": 1,
        "step": 1,
        "message": {
            "id": _system_id("v2", 1, "step/start"),
            "role": "system",
            "source": _system_source(),
            "content": [],
        },
    }
    promoted = events[3]
    assert promoted["time"] == 1002
    assert promoted["surfaceOp"] == {"op": "replace", "startSeq": 2, "endSeq": 2}
    assert promoted["sourceEventSeqs"] == [2]
    assert promoted["data"]["message"] == {
        "id": _system_id("v2", 2, "request/header"),
        "role": "system",
        "source": _system_source(),
        "content": [{"type": "text", "text": "Be helpful."}],
    }
    assert events[4]["data"]["header"] == {}


def test_v2_to_v3_prompt_change_and_clear() -> None:
    _, _, events = _run_stage(
        SESSION_FORMAT_V2_TO_V3,
        V2_HEADER,
        0,
        [
            *_v2_opening(),
            _v2_user(2, 1002, "m1", "hi"),
            _v2_request_header(3, 1003, "first"),
            _v2_request_header(4, 1004, "first"),
            _v2_request_header(5, 1005, "changed"),
            _v2_request_header(6, 1006, None),
            _v2_request_header(7, 1007, ""),
        ],
    )
    systems = [event for event in events if event["type"] == "system/message"]
    assert len(systems) == 4
    assert systems[0]["data"]["message"]["content"] == []
    assert systems[1]["data"]["message"]["content"] == [
        {"type": "text", "text": "first"}
    ]
    assert systems[2]["data"]["message"]["content"] == [
        {"type": "text", "text": "changed"}
    ]
    assert systems[3]["data"]["message"]["content"] == []
    assert all(
        "system" not in event["data"]["header"]
        for event in events
        if event["type"] == "request/header"
    )


def test_v2_to_v3_surface_before_first_step_refused() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="before first step"
    ):
        _run_stage(
            SESSION_FORMAT_V2_TO_V3,
            V2_HEADER,
            0,
            [_v2_user(0, 1000, "m1", "hi"), *_v2_opening()],
        )


def test_v2_to_v3_prompt_change_outside_step_refused() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="outside an open step"
    ):
        _run_stage(
            SESSION_FORMAT_V2_TO_V3,
            V2_HEADER,
            0,
            [
                {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
                _v2_request_header(1, 1001, "early"),
                {
                    "type": "step/start",
                    "seq": 2,
                    "time": 1002,
                    "data": {"turn": 1, "step": 1},
                },
            ],
        )


def test_v2_to_v3_ptc_renames() -> None:
    target_header, _, events = _run_stage(
        SESSION_FORMAT_V2_TO_V3,
        _header(2, "v2", agentPreset="code"),
        0,
        [
            *_v2_opening(),
            {
                "type": "tool/code-dispatch",
                "seq": 2,
                "time": 1002,
                "data": {
                    "rootCallId": "r",
                    "parentCallId": "p",
                    "subCallId": "s",
                    "name": "add",
                    "arguments": {},
                    "isError": False,
                    "content": [],
                },
            },
            {
                "type": "agent-preset/selected",
                "seq": 3,
                "time": 1003,
                "data": {"agentPreset": "code"},
            },
            {
                "type": "user/message",
                "seq": 4,
                "time": 1004,
                "data": {
                    "role": "user",
                    "id": "m1",
                    "content": [{"type": "text", "text": "hi"}],
                    "source": {"kind": "plugin", "plugin": "tools-code-mode"},
                },
                "surfaceOp": "append",
            },
        ],
    )
    assert target_header["agentPreset"] == "ptc"
    assert events[3]["type"] == "tool/ptc-dispatch"
    assert events[4]["data"] == {"agentPreset": "ptc"}
    assert events[5]["data"]["source"] == {"kind": "plugin", "plugin": "tools-ptc"}


def test_v2_to_v3_reference_remapping() -> None:
    _, _, events = _run_stage(
        SESSION_FORMAT_V2_TO_V3,
        V2_HEADER,
        0,
        [
            *_v2_opening(),
            _v2_user(2, 1002, "ma", "a"),
            _v2_request_header(3, 1003, "sys"),
            _v2_user(4, 1004, "mb", "b"),
            {
                "type": "compaction/prune",
                "seq": 5,
                "time": 1005,
                "data": {
                    "shadowedRange": {"start": 2, "end": 4},
                    "shadowedSeqs": [2, 4],
                    "shadowedTokenCount": 20,
                },
            },
            {
                **_v2_user(6, 1006, "m-rep", "replacement"),
                "surfaceOp": {"op": "replace", "start": 2, "end": 4},
                "sourceEventSeqs": [2, 4],
            },
            {
                "type": "command/run",
                "seq": 7,
                "time": 1007,
                "data": {"commandId": "c", "name": "x", "source": {"kind": "user"}},
            },
            {
                "type": "command/done",
                "seq": 8,
                "time": 1008,
                "data": {"commandId": "c", "kind": "success", "sourceEventSeq": 6},
            },
            {
                "type": "session/title",
                "seq": 9,
                "time": 1009,
                "data": {
                    "title": "title",
                    "messageSeqs": [2, 4],
                    "source": {"kind": "fallback"},
                },
            },
        ],
    )
    kept = [event for event in events if event["type"] != "system/message"]
    prune = kept[5]["data"]
    assert prune["shadowedRange"] == {"start": 3, "end": 6}
    assert prune["shadowedSeqs"] == [3, 6]
    assert kept[6]["surfaceOp"] == {"op": "replace", "startSeq": 3, "endSeq": 6}
    assert kept[6]["sourceEventSeqs"] == [3, 6]
    assert kept[8]["data"]["sourceEventSeq"] == 8
    assert kept[9]["data"]["messageSeqs"] == [3, 6]


def test_v2_to_v3_dense_seq_refused() -> None:
    with pytest.raises(SessionFormatError, match="must be dense"):
        _run_stage(
            SESSION_FORMAT_V2_TO_V3,
            V2_HEADER,
            0,
            [
                {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
                {
                    "type": "step/start",
                    "seq": 2,
                    "time": 1001,
                    "data": {"turn": 1, "step": 1},
                },
            ],
        )


def test_v2_to_v3_delivery_marker_foreign_refused() -> None:
    with pytest.raises(SessionFormatError, match="wrong Session"):
        _run_stage(
            SESSION_FORMAT_V2_TO_V3,
            V2_HEADER,
            0,
            [
                *_v2_opening(),
                {
                    "type": "session-log-deepseek/delivery-accepted",
                    "seq": 2,
                    "time": 1002,
                    "data": {
                        "sessionId": "foreign",
                        "throughSeq": 1,
                        "sessionFormatVersion": 2,
                    },
                },
            ],
        )


def test_v2_to_v3_seeded_cut_recovered() -> None:
    header = _header(2, "seeded", parentSession="parent", isSeeded=True)
    _, cut, events = _run_stage(
        SESSION_FORMAT_V2_TO_V3,
        header,
        4,
        [
            *_v2_opening(),
            _v2_request_header(2, 1002, "seed"),
            {
                "type": "session-log-deepseek/delivery-accepted",
                "seq": 3,
                "time": 1003,
                "data": {
                    "sessionId": "parent",
                    "throughSeq": 1,
                    "sessionFormatVersion": 2,
                },
            },
            {
                "type": "session/end-seed",
                "seq": 4,
                "time": 1004,
                "data": {"inherited": True},
            },
        ],
    )
    assert cut == 6
    assert events[6]["type"] == "session/end-seed"
    assert events[6]["data"] == {"inherited": True}


# ---------------------------------------------------------------------------
# Migration edge v3 -> v4
# ---------------------------------------------------------------------------

V3_HEADER = _header(3, "v3")


def _v3_migration(children: list | None = None) -> SessionFormatMigration:
    return create_session_format_v3_to_v4([] if children is None else children)


def _v3_tool_result(seq: int, time: int, **overrides: object) -> dict:
    message: dict = {
        "id": "m-tool-1",
        "role": "user",
        "source": {"kind": "tool", "callId": "c1"},
        "content": [
            {
                "type": "tool-result",
                "toolCallId": "c1",
                "content": [{"type": "text", "text": "3"}],
            }
        ],
    }
    message.update(overrides)
    return {
        "type": "tool/result",
        "seq": seq,
        "time": time,
        "data": {"turn": 1, "step": 1, "message": message},
        "surfaceOp": "append",
    }


def test_v3_to_v4_bare_edge_requires_child_facts() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="explicit historical child facts",
    ):
        SESSION_FORMAT_V3_TO_V4.create_stage(
            SessionFormatMigrationStageInput(
                source_header={
                    "version": 3,
                    "id": "v3",
                    "createdAt": 1000,
                    "isSeeded": False,
                    "delegationDepth": 0,
                },
                target_header={
                    "version": 4,
                    "id": "v3",
                    "createdAt": 1000,
                    "isSeeded": False,
                    "delegationDepth": 0,
                },
                source_inherited_event_count=0,
                source_kind="decoded",
            )
        )


def test_v3_to_v4_opaque_namespacing() -> None:
    _, _, events = _run_stage(
        _v3_migration(),
        V3_HEADER,
        0,
        [
            {
                "type": "vendor/widget",
                "seq": 0,
                "time": 1000,
                "data": {"x": 1},
                "ignorable": True,
            }
        ],
    )
    assert events[0]["type"] == "plugin:vendor/widget"
    assert events[0]["data"] == {"x": 1}
    assert events[0]["ignorable"] is True


def test_v3_to_v4_unknown_non_ignorable_refused() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="unknown event type"
    ):
        _run_stage(
            _v3_migration(),
            V3_HEADER,
            0,
            [{"type": "vendor/widget", "seq": 0, "time": 1000, "data": {"x": 1}}],
        )


def test_v3_to_v4_tool_result_lifted() -> None:
    _, _, events = _run_stage(
        _v3_migration(),
        V3_HEADER,
        0,
        [
            {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
            {
                "type": "step/start",
                "seq": 1,
                "time": 1001,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "tool/call",
                "seq": 2,
                "time": 1002,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "callId": "c1",
                    "name": "add",
                    "arguments": {"a": 1},
                },
            },
            _v3_tool_result(3, 1003),
            {
                "type": "step/end",
                "seq": 4,
                "time": 1004,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "turn/end",
                "seq": 5,
                "time": 1005,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
        ],
    )
    assert events[3]["data"]["message"] == {
        "role": "tool",
        "source": {"kind": "tool", "callId": "c1"},
        "toolCallId": "c1",
        "content": [{"type": "text", "text": "3"}],
        "id": "m-tool-1",
    }


def test_v3_to_v4_tool_result_nested_refused() -> None:
    with pytest.raises(SessionFormatError, match="tool-result"):
        _run_stage(
            _v3_migration(),
            V3_HEADER,
            0,
            [
                _v3_tool_result(
                    0,
                    1000,
                    content=[
                        {
                            "type": "tool-result",
                            "toolCallId": "c1",
                            "content": [
                                {
                                    "type": "tool-result",
                                    "toolCallId": "c1",
                                    "content": [],
                                }
                            ],
                        }
                    ],
                )
            ],
        )


def test_v3_to_v4_tool_result_call_mismatch_refused() -> None:
    with pytest.raises(SessionFormatError, match="matching its tool source"):
        _run_stage(
            _v3_migration(),
            V3_HEADER,
            0,
            [
                _v3_tool_result(
                    0,
                    1000,
                    content=[
                        {
                            "type": "tool-result",
                            "toolCallId": "c2",
                            "content": [{"type": "text", "text": "3"}],
                        }
                    ],
                )
            ],
        )


def test_v3_to_v4_source_rewrites() -> None:
    _, _, events = _run_stage(
        _v3_migration(),
        V3_HEADER,
        0,
        [
            {
                "type": "user/message",
                "seq": 0,
                "time": 1000,
                "data": {
                    "role": "user",
                    "id": "m1",
                    "content": [{"type": "text", "text": "a"}],
                    "source": {"kind": "plugin", "plugin": "compact"},
                },
                "surfaceOp": "append",
            },
            {
                "type": "assistant/message",
                "seq": 1,
                "time": 1001,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "message": {
                        "id": "m2",
                        "role": "assistant",
                        "source": {"kind": "plugin", "plugin": "tools-ptc"},
                        "content": [{"type": "text", "text": "b"}],
                    },
                    "stream": [],
                },
                "surfaceOp": "append",
            },
            {
                "type": "user/message",
                "seq": 2,
                "time": 1002,
                "data": {
                    "role": "user",
                    "id": "m3",
                    "content": [{"type": "text", "text": "c"}],
                    "source": {"kind": "plugin", "plugin": "vendor-x"},
                },
                "surfaceOp": "append",
            },
            {
                "type": "user/message",
                "seq": 3,
                "time": 1003,
                "data": {
                    "role": "user",
                    "id": "m4",
                    "content": [{"type": "text", "text": "d"}],
                    "source": {"kind": "goal"},
                },
                "surfaceOp": "append",
            },
        ],
    )
    assert events[0]["data"]["source"] == {"kind": "compact-checkpoint"}
    assert events[1]["data"]["message"]["source"] == {"kind": "ptc-mode"}
    assert events[2]["data"]["source"] == {"kind": "plugin:vendor-x"}
    assert events[3]["data"]["source"] == {"kind": "goal"}


def test_v3_to_v4_content_tag_namespaced() -> None:
    _, _, events = _run_stage(
        _v3_migration(),
        V3_HEADER,
        0,
        [
            {
                "type": "user/message",
                "seq": 0,
                "time": 1000,
                "data": {
                    "role": "user",
                    "id": "m1",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "widget", "data": {"x": 1}},
                    ],
                    "source": {"kind": "user"},
                },
                "surfaceOp": "append",
            }
        ],
    )
    assert events[0]["data"]["content"] == [
        {"type": "text", "text": "a"},
        {"type": "plugin:widget", "data": {"x": 1}},
    ]


def test_v3_to_v4_retired_tool_dispatch_refused() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="unknown event type"
    ):
        _run_stage(
            _v3_migration(),
            V3_HEADER,
            0,
            [
                {
                    "type": "tool/code-dispatch",
                    "seq": 0,
                    "time": 1000,
                    "data": {
                        "rootCallId": "r",
                        "parentCallId": "p",
                        "subCallId": "s",
                        "name": "add",
                        "arguments": {},
                        "isError": False,
                        "content": [],
                    },
                }
            ],
        )
    _, _, events = _run_stage(
        _v3_migration(),
        V3_HEADER,
        0,
        [
            {
                "type": "tool/code-dispatch",
                "seq": 0,
                "time": 1000,
                "data": {
                    "rootCallId": "r",
                    "parentCallId": "p",
                    "subCallId": "s",
                    "name": "add",
                    "arguments": {},
                    "isError": False,
                    "content": [],
                },
                "ignorable": True,
            }
        ],
    )
    assert events[0]["type"] == "plugin:tool/code-dispatch"


def test_v3_to_v4_defer_loading_refused() -> None:
    with pytest.raises(SessionFormatUnsupportedMigrationError, match="deferLoading"):
        _run_stage(
            _v3_migration(),
            V3_HEADER,
            0,
            [
                {
                    "type": "request/header",
                    "seq": 0,
                    "time": 1000,
                    "data": {
                        "header": {"tools": [{"name": "x", "deferLoading": True}]},
                        "reason": {"kind": "start"},
                    },
                }
            ],
        )


def test_v3_to_v4_child_facts_appended() -> None:
    children = [
        {
            "childId": "c1",
            "descriptor": {
                "version": 2,
                "provider": "p",
                "mode": "one-shot",
                "label": "L",
            },
            "descriptorCount": 1,
            "childCreatedAt": 500,
        }
    ]
    _, cut, events = _run_stage(
        _v3_migration(children),
        V3_HEADER,
        0,
        [{"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}}],
    )
    assert cut == 0
    assert events[1] == {
        "type": "subagent/catalog",
        "seq": 1,
        "time": 1000,
        "data": {
            "version": 0,
            "childId": "c1",
            "childCreatedAt": 500,
            "mode": "one-shot",
            "label": "L",
        },
    }


def test_v3_to_v4_child_fact_default_unknown() -> None:
    children = [
        {
            "childId": "c1",
            "descriptor": {"version": 9, "provider": "p"},
            "descriptorCount": 1,
            "childCreatedAt": 500,
        }
    ]
    _, _, events = _run_stage(
        _v3_migration(children),
        V3_HEADER,
        0,
        [{"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}}],
    )
    assert events[1]["data"] == {
        "version": 1,
        "childId": "c1",
        "childCreatedAt": 500,
        "mode": "unknown",
    }


def test_v3_to_v4_duplicate_catalog_child_refused() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="duplicate catalog child"
    ):
        _run_stage(
            _v3_migration(),
            V3_HEADER,
            0,
            [
                {
                    "type": "subagent/catalog",
                    "seq": 0,
                    "time": 1000,
                    "data": {
                        "version": 1,
                        "childId": "c1",
                        "childCreatedAt": 500,
                        "mode": "unknown",
                    },
                },
                {
                    "type": "subagent/catalog",
                    "seq": 1,
                    "time": 1001,
                    "data": {
                        "version": 1,
                        "childId": "c1",
                        "childCreatedAt": 500,
                        "mode": "unknown",
                    },
                },
            ],
        )


def test_v3_to_v4_catalog_conflict_refused() -> None:
    children = [
        {
            "childId": "c1",
            "descriptor": {
                "version": 2,
                "provider": "p",
                "mode": "continuable",
                "label": "L",
            },
            "descriptorCount": 1,
            "childCreatedAt": 500,
        }
    ]
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="conflicts with its parent catalog",
    ):
        _run_stage(
            _v3_migration(children),
            V3_HEADER,
            0,
            [
                {
                    "type": "subagent/catalog",
                    "seq": 0,
                    "time": 1000,
                    "data": {
                        "version": 1,
                        "childId": "c1",
                        "childCreatedAt": 500,
                        "mode": "one-shot",
                    },
                }
            ],
        )


def test_v3_to_v4_interrupted_turn_synthesized() -> None:
    _, _, events = _run_stage(
        _v3_migration(),
        V3_HEADER,
        0,
        [
            {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
            {
                "type": "agent/inbox/spliced",
                "seq": 1,
                "time": 1001,
                "data": {
                    "target": "next-turn",
                    "start": 0,
                    "inserted": [
                        {
                            "role": "user",
                            "id": "mx",
                            "content": [{"type": "text", "text": "x"}],
                            "source": {"kind": "user"},
                        }
                    ],
                },
            },
            {"type": "turn/start", "seq": 2, "time": 1002, "data": {"turn": 2}},
        ],
    )
    assert [event["type"] for event in events] == [
        "turn/start",
        "agent/inbox/spliced",
        "turn/end",
        "turn/start",
    ]
    assert events[2]["data"] == {"turn": 1, "reason": {"kind": "interrupted"}}


def test_v3_to_v4_delivery_v4_claim_refused() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="claims target format v4"
    ):
        _run_stage(
            _v3_migration(),
            V3_HEADER,
            0,
            [
                {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
                {
                    "type": "session-log-deepseek/delivery-accepted",
                    "seq": 1,
                    "time": 1001,
                    "data": {
                        "sessionId": "v3",
                        "throughSeq": 0,
                        "sessionFormatVersion": 4,
                    },
                },
            ],
        )


def test_child_catalog_fact_versions() -> None:
    v1 = child_catalog_fact(
        {
            "childId": "c1",
            "descriptor": {"version": 1, "provider": "p", "label": "L"},
            "descriptorCount": 1,
            "childCreatedAt": 500,
        }
    )
    assert v1 == {
        "version": 0,
        "childId": "c1",
        "childCreatedAt": 500,
        "mode": "continuable",
        "label": "L",
    }
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="invalid subagent descriptor mode"
    ):
        child_catalog_fact(
            {
                "childId": "c1",
                "descriptor": {"version": 2, "provider": "p", "mode": "bogus"},
                "descriptorCount": 1,
                "childCreatedAt": 500,
            }
        )
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="historical child identity"
    ):
        child_catalog_source(
            {"descriptor": {}, "descriptorCount": 1, "childCreatedAt": 1}
        )
    with pytest.raises(SessionFormatError, match="supported versioned catalog fact"):
        catalog_fact(
            {"version": 3, "childId": "c1", "childCreatedAt": 1, "mode": "unknown"}
        )


# ---------------------------------------------------------------------------
# End-to-end fixtures: v0/v1/v2/v3 -> byte-equivalent v4
# ---------------------------------------------------------------------------


def _v4_header(session_id: str) -> dict:
    return {
        "type": "session",
        "version": 4,
        "id": session_id,
        "createdAt": 1000,
        "isSeeded": False,
        "delegationDepth": 0,
    }


def _system_message(
    seq: int,
    time: int,
    session_id: str,
    anchor_seq: int,
    anchor_type: str,
    content: list,
) -> dict:
    return {
        "type": "system/message",
        "seq": seq,
        "time": time,
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "id": _system_id(session_id, anchor_seq, anchor_type),
                "role": "system",
                "source": {"kind": "system-prompt"},
                "content": content,
            },
        },
        "surfaceOp": "append",
    }


def test_e2e_v0_to_v4() -> None:
    artifact = _migrate(
        _header(0, "e2e-v0"),
        [
            {
                "type": "turn/start",
                "seq": 0,
                "time": 1000,
                "data": {"turn": 1, "trigger": {"kind": "user"}},
            },
            {
                "type": "step/start",
                "seq": 1,
                "time": 1001,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "steering/message",
                "seq": 2,
                "time": 1002,
                "data": {
                    "turn": 1,
                    "content": [{"type": "text", "text": "steer"}],
                    "source": {"kind": "user"},
                },
                "surfaceOp": "append",
            },
            {
                "type": "compact/start",
                "seq": 3,
                "time": 1003,
                "data": {"compactionId": "c1", "turn": 1},
            },
            {
                "type": "step/end",
                "seq": 4,
                "time": 1004,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "turn/end",
                "seq": 5,
                "time": 1005,
                "data": {"turn": 1, "reason": {"kind": "aborted"}},
            },
        ],
    )
    assert artifact.inherited_event_count == 0
    expected = [
        _v4_header("e2e-v0"),
        {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
        {"type": "step/start", "seq": 1, "time": 1001, "data": {"turn": 1, "step": 1}},
        _system_message(2, 1001, "e2e-v0", 1, "step/start", []),
        {
            "type": "user/message",
            "seq": 3,
            "time": 1002,
            "data": {
                "role": "user",
                "id": "legacy-message:e2e-v0:2",
                "content": [{"type": "text", "text": "steer"}],
                "source": {"kind": "user"},
            },
            "surfaceOp": "append",
        },
        {
            "type": "compaction/start",
            "seq": 4,
            "time": 1003,
            "data": {"compactionId": "c1", "turn": 1},
        },
        {"type": "step/end", "seq": 5, "time": 1004, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": 6,
            "time": 1005,
            "data": {
                "turn": 1,
                "reason": {"kind": "aborted", "reason": {"kind": "legacy"}},
            },
        },
    ]
    assert _encode_lines(artifact) == [_dump(line) for line in expected]


def test_e2e_v1_to_v4() -> None:
    artifact = _migrate(
        _header(1, "e2e-v1"),
        [
            {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
            {
                "type": "step/start",
                "seq": 1,
                "time": 1001,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "user/message",
                "seq": 2,
                "time": 1002,
                "data": {
                    "role": "user",
                    "id": "m1",
                    "content": [{"type": "text", "text": "hi"}],
                    "source": {"kind": "user"},
                },
                "surfaceOp": "append",
            },
            {
                "type": "assistant/chunk",
                "seq": 3,
                "time": 1003,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "chunk": {"type": "text-delta", "index": 0, "text": "Hello"},
                },
            },
            {
                "type": "assistant/chunk",
                "seq": 4,
                "time": 1004,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "chunk": {"type": "text-delta", "index": 0, "text": " world"},
                },
            },
            {
                "type": "assistant/message",
                "seq": 5,
                "time": 1005,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "message": {
                        "id": "m2",
                        "role": "assistant",
                        "source": {"kind": "model"},
                        "content": [{"type": "text", "text": "Hello world"}],
                    },
                },
                "sourceEventSeqs": [3, 4],
                "surfaceOp": "append",
            },
            {
                "type": "step/end",
                "seq": 6,
                "time": 1006,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "turn/end",
                "seq": 7,
                "time": 1007,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
        ],
    )
    assert artifact.inherited_event_count == 0
    expected = [
        _v4_header("e2e-v1"),
        {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
        {"type": "step/start", "seq": 1, "time": 1001, "data": {"turn": 1, "step": 1}},
        _system_message(2, 1001, "e2e-v1", 1, "step/start", []),
        {
            "type": "user/message",
            "seq": 3,
            "time": 1002,
            "data": {
                "role": "user",
                "id": "m1",
                "content": [{"type": "text", "text": "hi"}],
                "source": {"kind": "user"},
            },
            "surfaceOp": "append",
        },
        {
            "type": "assistant/message",
            "seq": 4,
            "time": 1005,
            "data": {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "m2",
                    "role": "assistant",
                    "source": {"kind": "model"},
                    "content": [{"type": "text", "text": "Hello world"}],
                },
                "stream": [
                    {
                        "type": "text-chunks",
                        "index": 0,
                        "time0": 1003,
                        "dt": [1],
                        "texts": ["Hello", " world"],
                    }
                ],
            },
            "surfaceOp": "append",
        },
        {"type": "step/end", "seq": 5, "time": 1006, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": 6,
            "time": 1007,
            "data": {"turn": 1, "reason": {"kind": "done"}},
        },
    ]
    assert _encode_lines(artifact) == [_dump(line) for line in expected]


def test_e2e_v2_to_v4() -> None:
    artifact = _migrate(
        _header(2, "e2e-v2"),
        [
            {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
            {
                "type": "step/start",
                "seq": 1,
                "time": 1001,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "request/header",
                "seq": 2,
                "time": 1002,
                "data": {
                    "header": {"system": "Be helpful."},
                    "reason": {"kind": "start"},
                },
            },
            {
                "type": "user/message",
                "seq": 3,
                "time": 1003,
                "data": {
                    "role": "user",
                    "id": "m1",
                    "content": [{"type": "text", "text": "hi"}],
                    "source": {"kind": "user"},
                },
                "surfaceOp": "append",
            },
            {
                "type": "step/end",
                "seq": 4,
                "time": 1004,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "turn/end",
                "seq": 5,
                "time": 1005,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
        ],
    )
    assert artifact.inherited_event_count == 0
    promoted = _system_message(
        3,
        1002,
        "e2e-v2",
        2,
        "request/header",
        [{"type": "text", "text": "Be helpful."}],
    )
    promoted["surfaceOp"] = {"op": "replace", "startSeq": 2, "endSeq": 2}
    promoted["sourceEventSeqs"] = [2]
    expected = [
        _v4_header("e2e-v2"),
        {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
        {"type": "step/start", "seq": 1, "time": 1001, "data": {"turn": 1, "step": 1}},
        _system_message(2, 1001, "e2e-v2", 1, "step/start", []),
        promoted,
        {
            "type": "request/header",
            "seq": 4,
            "time": 1002,
            "data": {"header": {}, "reason": {"kind": "start"}},
        },
        {
            "type": "user/message",
            "seq": 5,
            "time": 1003,
            "data": {
                "role": "user",
                "id": "m1",
                "content": [{"type": "text", "text": "hi"}],
                "source": {"kind": "user"},
            },
            "surfaceOp": "append",
        },
        {"type": "step/end", "seq": 6, "time": 1004, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": 7,
            "time": 1005,
            "data": {"turn": 1, "reason": {"kind": "done"}},
        },
    ]
    assert _encode_lines(artifact) == [_dump(line) for line in expected]


def test_e2e_v3_to_v4() -> None:
    artifact = _migrate(
        _header(3, "e2e-v3"),
        [
            {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
            {
                "type": "step/start",
                "seq": 1,
                "time": 1001,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "user/message",
                "seq": 2,
                "time": 1002,
                "data": {
                    "role": "user",
                    "id": "m-user-1",
                    "content": [{"type": "text", "text": "hello"}],
                    "source": {"kind": "user"},
                },
                "surfaceOp": "append",
            },
            {
                "type": "assistant/message",
                "seq": 3,
                "time": 1003,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "message": {
                        "id": "m-asst-1",
                        "role": "assistant",
                        "source": {"kind": "model"},
                        "content": [{"type": "text", "text": "hi there"}],
                    },
                    "stream": [],
                },
                "surfaceOp": "append",
            },
            {
                "type": "tool/call",
                "seq": 4,
                "time": 1004,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "callId": "c1",
                    "name": "add",
                    "arguments": {"a": 1},
                },
            },
            {
                "type": "tool/result",
                "seq": 5,
                "time": 1005,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "message": {
                        "id": "m-tool-1",
                        "role": "user",
                        "source": {"kind": "tool", "callId": "c1"},
                        "content": [
                            {
                                "type": "tool-result",
                                "toolCallId": "c1",
                                "content": [{"type": "text", "text": "3"}],
                            }
                        ],
                    },
                },
                "surfaceOp": "append",
            },
            {
                "type": "step/end",
                "seq": 6,
                "time": 1006,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "turn/end",
                "seq": 7,
                "time": 1007,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
        ],
    )
    assert artifact.inherited_event_count == 0
    expected = [
        _v4_header("e2e-v3"),
        {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
        {"type": "step/start", "seq": 1, "time": 1001, "data": {"turn": 1, "step": 1}},
        {
            "type": "user/message",
            "seq": 2,
            "time": 1002,
            "data": {
                "role": "user",
                "id": "m-user-1",
                "content": [{"type": "text", "text": "hello"}],
                "source": {"kind": "user"},
            },
            "surfaceOp": "append",
        },
        {
            "type": "assistant/message",
            "seq": 3,
            "time": 1003,
            "data": {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "m-asst-1",
                    "role": "assistant",
                    "source": {"kind": "model"},
                    "content": [{"type": "text", "text": "hi there"}],
                },
                "stream": [],
            },
            "surfaceOp": "append",
        },
        {
            "type": "tool/call",
            "seq": 4,
            "time": 1004,
            "data": {
                "turn": 1,
                "step": 1,
                "callId": "c1",
                "name": "add",
                "arguments": {"a": 1},
            },
        },
        {
            "type": "tool/result",
            "seq": 5,
            "time": 1005,
            "data": {
                "turn": 1,
                "step": 1,
                "message": {
                    "role": "tool",
                    "source": {"kind": "tool", "callId": "c1"},
                    "toolCallId": "c1",
                    "content": [{"type": "text", "text": "3"}],
                    "id": "m-tool-1",
                },
            },
            "surfaceOp": "append",
        },
        {"type": "step/end", "seq": 6, "time": 1006, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": 7,
            "time": 1007,
            "data": {"turn": 1, "reason": {"kind": "done"}},
        },
    ]
    assert _encode_lines(artifact) == [_dump(line) for line in expected]


def test_e2e_seeded_cut_from_end_seed_marker() -> None:
    artifact = _migrate(
        _header(3, "seeded-v3", isSeeded=True),
        [
            {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
            {
                "type": "step/start",
                "seq": 1,
                "time": 1001,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "session/end-seed",
                "seq": 2,
                "time": 1002,
                "data": {"inherited": True},
            },
            {
                "type": "user/message",
                "seq": 3,
                "time": 1003,
                "data": {
                    "role": "user",
                    "id": "m1",
                    "content": [{"type": "text", "text": "hi"}],
                    "source": {"kind": "user"},
                },
                "surfaceOp": "append",
            },
            {
                "type": "step/end",
                "seq": 4,
                "time": 1004,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "turn/end",
                "seq": 5,
                "time": 1005,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
        ],
    )
    # The inherited cut comes from the tagged marker, not from any header
    # field: the header carries no cut at all.
    assert artifact.inherited_event_count == 2
    assert artifact.events[2]["type"] == "session/end-seed"


def test_e2e_newer_version_refused_without_row_decode() -> None:
    header = _header(9, "future")
    result = SESSION_FORMAT_CATALOG.read_header(header)
    assert result.status == "unsupported"
    assert result.reason == "stored Session uses newer format v9; this build writes v4"
    # The stable user-facing upgrade text is produced by
    # session_format_version_refusal and surfaced by the load path; the
    # catalog itself refuses with its internal newer-format text.
    assert session_format_version_refusal("future", 9) == (
        'session "future" uses log format v9, but this harness reads only v4: '
        "the log was written by a newer harness — upgrade the harness to open it"
    )
    with pytest.raises(SessionFormatUnsupportedMigrationError, match="newer format v9"):
        SESSION_FORMAT_CATALOG.create_restore(header)


def test_e2e_corrupt_row_refused_at_format_level() -> None:
    restore = SESSION_FORMAT_CATALOG.create_restore(
        _header(4, "s"), validation="current", recovery="strict"
    )
    with pytest.raises(SessionFormatError, match="must be a JSON object"):
        restore.decode_row("not an object")
    restore = SESSION_FORMAT_CATALOG.create_restore(
        _header(4, "s"), validation="current", recovery="strict"
    )
    restore.decode_row(
        {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}}
    )
    with pytest.raises(SessionFormatError, match="seq gap"):
        restore.decode_row(
            {"type": "turn/start", "seq": 2, "time": 1001, "data": {"turn": 2}}
        )


# ---------------------------------------------------------------------------
# Boundary helpers: seq ranges, filenames, envelopes, watermarks
# ---------------------------------------------------------------------------


def test_require_safe_integer_and_exact_keys() -> None:
    assert require_safe_integer(5, "x") == 5
    assert require_safe_integer(-3, "x") == -3
    with pytest.raises(SessionFormatError, match="must be a safe integer"):
        require_safe_integer("5", "x")
    with pytest.raises(SessionFormatError, match="must be a safe integer"):
        require_safe_integer(True, "x")
    with pytest.raises(SessionFormatError, match="has unexpected member"):
        exact_keys({"a": 1, "b": 2}, ["a"], [], "label")
    with pytest.raises(SessionFormatError, match="lacks required member"):
        exact_keys({"a": 1}, ["a", "b"], [], "label")
    exact_keys({"a": 1}, ["a"], ["b"], "label")


def test_snapshot_header() -> None:
    header = {
        "version": 4,
        "id": "s",
        "createdAt": 1,
        "isSeeded": False,
        "delegationDepth": 0,
    }
    assert snapshot_header(header) is header
    with pytest.raises(SessionFormatError, match="must be a JSON object"):
        snapshot_header("nope")
    with pytest.raises(SessionFormatError, match="id must be a string"):
        snapshot_header({**header, "id": 3})
    with pytest.raises(SessionFormatError, match="isSeeded must be a boolean"):
        snapshot_header({**header, "isSeeded": "yes"})
    with pytest.raises(SessionFormatError, match="createdAt"):
        snapshot_header({**header, "createdAt": -1})


def test_parse_session_format_log_filename_edges() -> None:
    assert parse_session_format_log_filename("session.jsonl") == 0
    assert parse_session_format_log_filename("session.v4.jsonl") == 4
    assert parse_session_format_log_filename("session.v10.jsonl") == 10
    assert parse_session_format_log_filename("session.v0.jsonl") is None
    assert parse_session_format_log_filename("session.v.jsonl") is None
    assert parse_session_format_log_filename("session.v04.jsonl") is None
    assert parse_session_format_log_filename("session.V4.jsonl") is None
    assert parse_session_format_log_filename("session.v4.jsonl.gz") is None
    assert parse_session_format_log_filename("other.jsonl") is None


def test_encode_seq_ranges_shapes() -> None:
    assert encode_seq_ranges([]) == []
    assert encode_seq_ranges([0, 1]) == [0, 1]
    assert encode_seq_ranges([0, 1, 2]) == [[0, 2]]
    assert encode_seq_ranges([0, 1, 2, 5]) == [[0, 2], 5]
    assert encode_seq_ranges([0, 2, 3]) == [0, 2, 3]
    assert encode_seq_ranges([3, 2]) == [3, 2]
    assert encode_seq_ranges([0, 1, 2, 5, 6, 7, 8, 10]) == [[0, 2], [5, 8], 10]


def test_decode_seq_ranges_errors() -> None:
    with pytest.raises(SessionFormatError, match="must be an array"):
        decode_seq_ranges("nope")
    with pytest.raises(SessionFormatError, match="exceeds its event seq"):
        decode_seq_ranges([0, 1, 2], max_entries=2)
    with pytest.raises(SessionFormatError, match="must be a \\[start, end\\] pair"):
        decode_seq_ranges([[0, 1, 2]])
    with pytest.raises(SessionFormatError, match="exceeds its event seq"):
        decode_seq_ranges([[2, 1]])
    with pytest.raises(SessionFormatError, match="exceeds its event seq"):
        decode_seq_ranges([[0, 5]], max_entries=3)
    with pytest.raises(SessionFormatError, match="unique earlier seqs"):
        decode_seq_ranges([1, [0, 1]], max_entries=5)
    with pytest.raises(SessionFormatError, match="strictly increasing"):
        decode_seq_ranges([[2, 3], 1])
    assert decode_seq_ranges([[0, 2], 5]) == [0, 1, 2, 5]
    assert decode_seq_ranges([3, 1]) == [3, 1]


def test_decode_envelope_row_errors() -> None:
    base = {"type": "user/message", "seq": 1, "time": 100, "data": {}}
    with pytest.raises(SessionFormatError, match="lacks required field"):
        decode_envelope_row({k: v for k, v in base.items() if k != "time"}, 0, "s")
    with pytest.raises(SessionFormatError, match="has unexpected field"):
        decode_envelope_row({**base, "extra": 1}, 0, "s")
    with pytest.raises(SessionFormatError, match="type must be a string"):
        decode_envelope_row({**base, "type": 7}, 0, "s")
    with pytest.raises(SessionFormatError, match="ignorable must be true"):
        decode_envelope_row({**base, "ignorable": False}, 0, "s")
    decoded = decode_envelope_row(
        {**base, "seq": 5, "sourceEventSeqs": [[0, 1]]}, 1, "s"
    )
    assert decoded["sourceEventSeqs"] == [0, 1]
    assert encode_envelope_event({**base, "sourceEventSeqs": [0, 1, 2, 5]})[
        "sourceEventSeqs"
    ] == [[0, 2], 5]


def test_validate_delivery_accepted_empty_id() -> None:
    with pytest.raises(SessionFormatError, match="nonempty Session id"):
        validate_delivery_accepted(
            {
                "type": "session-log-deepseek/delivery-accepted",
                "seq": 5,
                "time": 1,
                "data": {"sessionId": "", "throughSeq": 1, "sessionFormatVersion": 3},
            },
            3,
        )


def test_accepted_through_malformed() -> None:
    with pytest.raises(SessionFormatError, match="malformed acceptance format version"):
        accepted_through(
            [
                {
                    "type": "session-log-deepseek/delivery-accepted",
                    "seq": 5,
                    "time": 1,
                    "data": {
                        "sessionId": "s",
                        "throughSeq": 1,
                        "sessionFormatVersion": "3",
                    },
                }
            ],
            3,
            "s",
        )
    with pytest.raises(SessionFormatError, match="malformed acceptance watermark"):
        accepted_through(
            [
                {
                    "type": "session-log-deepseek/delivery-accepted",
                    "seq": 5,
                    "time": 1,
                    "data": {
                        "sessionId": "s",
                        "throughSeq": "1",
                        "sessionFormatVersion": 3,
                    },
                }
            ],
            3,
            "s",
        )
    with pytest.raises(SessionFormatError, match="malformed acceptance"):
        accepted_through(
            [
                {
                    "type": "session-log-deepseek/delivery-accepted",
                    "seq": 5,
                    "time": 1,
                    "data": {
                        "sessionId": "",
                        "throughSeq": 1,
                        "sessionFormatVersion": 3,
                    },
                }
            ],
            3,
            "s",
        )
    with pytest.raises(SessionFormatError, match="malformed acceptance"):
        accepted_through(
            [
                {
                    "type": "session-log-deepseek/delivery-accepted",
                    "seq": 5,
                    "time": 1,
                    "data": {
                        "sessionId": "s",
                        "throughSeq": 5,
                        "sessionFormatVersion": 3,
                    },
                }
            ],
            3,
            "s",
        )


def test_codec_encode_header_variants() -> None:
    v2 = _V2Codec()
    with pytest.raises(SessionFormatError, match="unseeded.*inherited events"):
        v2.encode_header(
            {
                "version": 2,
                "id": "s",
                "createdAt": 1,
                "isSeeded": False,
                "delegationDepth": 0,
            },
            3,
        )
    assert (
        v2.encode_header(
            {
                "version": 2,
                "id": "s",
                "createdAt": 1,
                "isSeeded": True,
                "delegationDepth": 0,
            },
            2,
        )["version"]
        == 2
    )
    v3 = _V3Codec()
    encoded = v3.encode_header(
        {
            "version": 3,
            "id": "s",
            "createdAt": 1,
            "isSeeded": False,
            "delegationDepth": 0,
        },
        0,
    )
    assert encoded["version"] == 3
    assert encoded["type"] == "session"
    v4 = _V4Codec()
    assert (
        v4.encode_header(
            {
                "version": 4,
                "id": "s",
                "createdAt": 1,
                "isSeeded": False,
                "delegationDepth": 0,
            },
            0,
        )["version"]
        == 4
    )


# ---------------------------------------------------------------------------
# Accumulator, packed runs, physical decoders
# ---------------------------------------------------------------------------


def test_accumulator_merges_text_and_reasoning_deltas() -> None:
    acc = AssistantStreamAccumulator()
    acc.push(100, {"type": "text-delta", "index": 0, "text": "Hello"})
    acc.push(101, {"type": "text-delta", "index": 0, "text": " world"})
    acc.push(105, {"type": "text-delta", "index": 1, "text": "!"})
    acc.push(106, {"type": "reasoning-delta", "index": 0, "text": "think"})
    acc.push(107, {"type": "reasoning-delta", "index": 0, "text": " more"})
    assert acc.snapshot() == [
        {
            "type": "text-chunks",
            "time0": 100,
            "index": 0,
            "dt": [1],
            "texts": ["Hello", " world"],
        },
        {"type": "text-chunks", "time0": 105, "index": 1, "dt": [], "texts": ["!"]},
        {
            "type": "reasoning-chunks",
            "time0": 106,
            "index": 0,
            "dt": [1],
            "texts": ["think", " more"],
        },
    ]


def test_accumulator_raw_records_and_errors() -> None:
    acc = AssistantStreamAccumulator()
    acc.push(100, {"type": "block-start", "index": 0})
    acc.push(101, {"type": "block-end", "index": 0})
    acc.push(102, {"type": "usage", "inputTokens": 1, "outputTokens": 2})
    acc.push(103, {"type": "finish", "reason": {"kind": "stop"}})
    snapshot = acc.snapshot()
    assert [record["type"] for record in snapshot] == [
        "chunk",
        "chunk",
        "chunk",
        "chunk",
    ]
    assert snapshot[0]["chunk"] == {"type": "block-start", "index": 0}
    with pytest.raises(SessionFormatError, match="index must be an integer"):
        acc.push(104, {"type": "text-delta", "index": "0", "text": "x"})
    with pytest.raises(SessionFormatError, match="text must be a string"):
        acc.push(104, {"type": "text-delta", "index": 0, "text": 3})
    with pytest.raises(SessionFormatError, match="unknown stream chunk type"):
        acc.push(104, {"type": "sparkle", "index": 0})
    with pytest.raises(SessionFormatError, match="must be a safe integer"):
        acc.push(2**60, {"type": "text-delta", "index": 0, "text": "x"})


def test_accumulator_unsafe_gap_starts_new_record() -> None:
    acc = AssistantStreamAccumulator()
    acc.push(-(2**53 - 1), {"type": "text-delta", "index": 0, "text": "a"})
    acc.push(2**53 - 1, {"type": "text-delta", "index": 0, "text": "b"})
    snapshot = acc.snapshot()
    assert len(snapshot) == 2
    assert snapshot[0]["texts"] == ["a"]
    assert snapshot[1]["texts"] == ["b"]


def test_accumulator_tool_call_deltas() -> None:
    acc = AssistantStreamAccumulator()
    acc.push(
        100,
        {
            "type": "tool-call-delta",
            "index": 0,
            "id": "c1",
            "name": "add",
            "argumentsDelta": "{",
        },
    )
    acc.push(
        101,
        {
            "type": "tool-call-delta",
            "index": 0,
            "id": "c1",
            "name": "add",
            "argumentsDelta": "}",
        },
    )
    acc.push(
        102,
        {
            "type": "tool-call-delta",
            "index": 0,
            "id": "c2",
            "name": "add",
            "argumentsDelta": "{}",
        },
    )
    acc.push(
        103,
        {
            "type": "tool-call-delta",
            "index": 0,
            "id": "",
            "name": "add",
            "argumentsDelta": "{}",
        },
    )
    acc.push(
        104,
        {
            "type": "tool-call-delta",
            "index": 0,
            "id": "c3",
            "name": "",
            "argumentsDelta": "{}",
        },
    )
    acc.push(
        105, {"type": "tool-call-delta", "index": 0, "id": "c4", "argumentsDelta": "{}"}
    )
    snapshot = acc.snapshot()
    assert snapshot[0] == {
        "type": "tool-call-chunks",
        "time0": 100,
        "index": 0,
        "dt": [1],
        "id": "c1",
        "name": "add",
        "args": ["{", "}"],
    }
    assert snapshot[1]["id"] == "c2"
    assert snapshot[2] == {
        "type": "chunk",
        "time": 103,
        "chunk": {
            "type": "tool-call-delta",
            "index": 0,
            "id": "",
            "name": "add",
            "argumentsDelta": "{}",
        },
    }
    assert snapshot[3]["type"] == "chunk"
    assert snapshot[4] == {
        "type": "tool-call-chunks",
        "time0": 105,
        "index": 0,
        "dt": [],
        "id": "c4",
        "args": ["{}"],
    }
    with pytest.raises(SessionFormatError, match="index must be an integer"):
        acc.push(
            106,
            {
                "type": "tool-call-delta",
                "index": "0",
                "id": "c",
                "argumentsDelta": "{}",
            },
        )
    with pytest.raises(SessionFormatError, match="id must be a string"):
        acc.push(
            106,
            {"type": "tool-call-delta", "index": 0, "id": 3, "argumentsDelta": "{}"},
        )
    with pytest.raises(SessionFormatError, match="name must be a string"):
        acc.push(
            106,
            {
                "type": "tool-call-delta",
                "index": 0,
                "id": "c",
                "name": 3,
                "argumentsDelta": "{}",
            },
        )
    with pytest.raises(SessionFormatError, match="argumentsDelta must be a string"):
        acc.push(
            106, {"type": "tool-call-delta", "index": 0, "id": "c", "argumentsDelta": 3}
        )


def test_packed_run_expand_variants() -> None:
    reasoning = PackedAssistantChunkRun(
        run_type="released-assistant-chunks",
        first_seq=0,
        event_count=2,
        stream={
            "type": "reasoning-chunks",
            "index": 0,
            "time0": 5,
            "dt": [2],
            "texts": ["a", "b"],
        },
        turn=1,
        step=1,
        last_time=7,
    )
    expanded = reasoning.expand()
    assert expanded[0]["data"]["chunk"] == {
        "type": "reasoning-delta",
        "index": 0,
        "text": "a",
    }
    assert expanded[1]["time"] == 7
    tool = PackedAssistantChunkRun(
        run_type="released-assistant-chunks",
        first_seq=3,
        event_count=1,
        stream={
            "type": "tool-call-chunks",
            "index": 0,
            "time0": 9,
            "dt": [],
            "id": "c1",
            "args": ["{}"],
        },
        turn=1,
        step=1,
        last_time=9,
    )
    assert tool.expand()[0]["data"]["chunk"] == {
        "type": "tool-call-delta",
        "index": 0,
        "id": "c1",
        "argumentsDelta": "{}",
    }


def _v0_decoder(recovery: str = "strict", **header_extra: object) -> object:
    return _V01Codec(0).create_decoder(_header(0, "s", **header_extra), recovery)


def _v2_decoder(recovery: str = "strict", **header_extra: object) -> object:
    return _V2Codec().create_decoder(_header(2, "s", **header_extra), recovery)


def test_v01_decoder_packed_runs() -> None:
    decoder = _v0_decoder()
    out = _Collector()
    decoder.decode_row(
        {
            "type": "text-chunks",
            "seq0": 0,
            "time0": 100,
            "data": {"turn": 1, "step": 1, "index": 0, "dt": [1], "texts": ["a", "b"]},
        },
        out,
    )
    assert [event["type"] for event in out.values] == [
        "assistant/chunk",
        "assistant/chunk",
    ]
    assert out.values[0]["data"]["chunk"] == {
        "type": "text-delta",
        "index": 0,
        "text": "a",
    }
    assert out.values[1]["time"] == 101
    decoder.decode_row(
        {
            "type": "tool-call-chunks",
            "seq0": 2,
            "time0": 200,
            "data": {
                "turn": 1,
                "step": 1,
                "index": 0,
                "id": "c1",
                "name": "add",
                "dt": [],
                "args": ["{}"],
            },
        },
        out,
    )
    assert out.values[2]["data"]["chunk"]["type"] == "tool-call-delta"
    assert decoder.finish() == 0


def test_v01_decoder_packed_run_errors() -> None:
    decoder = _v0_decoder()
    out = _Collector()
    with pytest.raises(SessionFormatError, match="non-empty string array"):
        decoder.decode_row(
            {
                "type": "text-chunks",
                "seq0": 0,
                "time0": 100,
                "data": {"turn": 1, "step": 1, "index": 0, "dt": [], "texts": []},
            },
            out,
        )
    with pytest.raises(SessionFormatError, match="dt length must match"):
        decoder.decode_row(
            {
                "type": "text-chunks",
                "seq0": 0,
                "time0": 100,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "index": 0,
                    "dt": [1, 2],
                    "texts": ["a", "b"],
                },
            },
            out,
        )
    with pytest.raises(SessionFormatError, match="id and optional name"):
        decoder.decode_row(
            {
                "type": "tool-call-chunks",
                "seq0": 0,
                "time0": 100,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "index": 0,
                    "id": "",
                    "dt": [],
                    "args": ["{}"],
                },
            },
            out,
        )
    with pytest.raises(SessionFormatError, match="has unexpected member"):
        decoder.decode_row(
            {
                "type": "text-chunks",
                "seq0": 0,
                "time0": 100,
                "extra": 1,
                "data": {"turn": 1, "step": 1, "index": 0, "dt": [], "texts": ["a"]},
            },
            out,
        )


def test_v01_decoder_recoverable_semantics() -> None:
    decoder = _v0_decoder("recoverable")
    out = _Collector()
    decoder.decode_row("not an object", out)
    assert out.values == []
    decoder.decode_row(
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}, out
    )
    assert out.values == []
    with pytest.raises(SessionFormatError, match="must be a JSON object"):
        decoder.decode_row(
            {
                "type": "turn/end",
                "seq": 1,
                "time": 101,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
            out,
        )
    strict = _v0_decoder("strict")
    with pytest.raises(SessionFormatError, match="must be a JSON object"):
        strict.decode_row("not an object", _Collector())


def test_v01_decoder_malformed_non_json_row() -> None:
    """Verifies a programmatic non-JSON row reaches the malformed-row handler.

    JSON rows cannot reach it (their keys are always strings), but decode_row
    takes Any and is a catalog entry point, so a packed row carrying a
    non-serializable key fails inside exact_keys and must surface as a
    malformed row instead of escaping the handler.
    """
    row: dict = {
        "type": "text-chunks",
        "seq0": 0,
        "time0": 100,
        object(): 1,
        "data": {"turn": 1, "step": 1, "index": 0, "dt": [], "texts": ["a"]},
    }
    strict = _v0_decoder("strict")
    with pytest.raises(
        SessionFormatError, match="released Session row 0 is malformed"
    ) as excinfo:
        strict.decode_row(row, _Collector())
    assert isinstance(excinfo.value.__cause__, TypeError)

    recoverable = _v0_decoder("recoverable")
    out = _Collector()
    recoverable.decode_row(dict(row), out)
    recoverable.decode_row(dict(row), out)
    assert out.values == []
    with pytest.raises(SessionFormatError, match="released Session row 0 is malformed"):
        recoverable.decode_row(
            {
                "type": "turn/end",
                "seq": 0,
                "time": 101,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
            out,
        )


def test_v01_decoder_seq_gap_and_finish() -> None:
    strict = _v0_decoder("strict")
    out = _Collector()
    strict.decode_row(
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}, out
    )
    with pytest.raises(SessionFormatError, match="seq gap"):
        strict.decode_row(
            {"type": "turn/start", "seq": 2, "time": 101, "data": {"turn": 2}}, out
        )
    recoverable = _v0_decoder("recoverable")
    out = _Collector()
    recoverable.decode_row(
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}, out
    )
    recoverable.decode_row(
        {"type": "turn/start", "seq": 2, "time": 101, "data": {"turn": 2}}, out
    )
    assert len(out.values) == 1
    with pytest.raises(SessionFormatError, match="seq gap"):
        recoverable.decode_row(
            {
                "type": "turn/end",
                "seq": 3,
                "time": 102,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
            out,
        )
    seeded = _v0_decoder("strict", seedLength=5)
    out = _Collector()
    seeded.decode_row(
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}, out
    )
    with pytest.raises(SessionFormatError, match="exceeds its event count"):
        seeded.finish()


def test_v01_codec_version_check() -> None:
    with pytest.raises(SessionFormatError, match="requires version 0 or 1"):
        _V01Codec(2)


def test_v2_decoder_recoverable_and_cut() -> None:
    strict = _v2_decoder("strict")
    out = _Collector()
    with pytest.raises(SessionFormatError, match="lacks required field"):
        strict.decode_row({"type": "turn/start", "seq": 0, "data": {"turn": 1}}, out)
    recoverable = _v2_decoder("recoverable")
    out = _Collector()
    recoverable.decode_row({"type": "turn/start", "seq": 0, "data": {"turn": 1}}, out)
    assert out.values == []
    recoverable.decode_row(
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}, out
    )
    assert out.values == []
    with pytest.raises(SessionFormatError, match="lacks required field"):
        recoverable.decode_row(
            {
                "type": "turn/end",
                "seq": 1,
                "time": 101,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
            out,
        )
    gap = _v2_decoder("strict")
    out = _Collector()
    gap.decode_row(
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}, out
    )
    with pytest.raises(SessionFormatError, match="seq gap"):
        gap.decode_row(
            {"type": "turn/start", "seq": 2, "time": 101, "data": {"turn": 2}}, out
        )


def test_v2_decoder_end_seed_agreement() -> None:
    seeded = _v2_decoder("strict", isSeeded=True)
    out = _Collector()
    seeded.decode_row(
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}, out
    )
    seeded.decode_row(
        {
            "type": "session/end-seed",
            "seq": 1,
            "time": 101,
            "data": {"inherited": True},
        },
        out,
    )
    assert seeded.finish() == 1
    no_marker = _v2_decoder("strict", isSeeded=True)
    no_marker.decode_row(
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}, _Collector()
    )
    with pytest.raises(SessionFormatError, match="lacks an inherited end-seed marker"):
        no_marker.finish()
    unseeded = _v2_decoder("strict")
    out = _Collector()
    unseeded.decode_row(
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}, out
    )
    unseeded.decode_row(
        {
            "type": "session/end-seed",
            "seq": 1,
            "time": 101,
            "data": {"inherited": True},
        },
        out,
    )
    with pytest.raises(SessionFormatError, match="unseeded.*inherited end-seed marker"):
        unseeded.finish()


def test_v2_physical_header_errors() -> None:
    base = {
        "type": "session",
        "version": 2,
        "id": "s",
        "createdAt": 1,
        "isSeeded": False,
        "delegationDepth": 0,
    }
    with pytest.raises(SessionFormatError, match="expected released v2"):
        _V2Codec().decode_header({**base, "version": 3})
    with pytest.raises(SessionFormatError, match="id must be a string"):
        _V2Codec().decode_header({**base, "id": 7})
    with pytest.raises(SessionFormatError, match="isSeeded must be boolean"):
        _V2Codec().decode_header({**base, "isSeeded": "yes"})
    with pytest.raises(SessionFormatError, match="must be a string"):
        _V2Codec().decode_header({**base, "cwd": 7})
    with pytest.raises(SessionFormatError, match='origin must be "subagent"'):
        _V2Codec().decode_header({**base, "origin": "user"})


def test_v3_decoder_version_and_admission() -> None:
    with pytest.raises(SessionFormatError, match="expected format v3"):
        _V3Codec().create_decoder(_header(4, "s"), "strict")
    decoder = _V3Codec().create_decoder(_header(3, "s"), "strict")
    out = _Collector()
    with pytest.raises(SessionFormatError, match="must be a JSON object"):
        decoder.decode_row("not an object", out)
    with pytest.raises(SessionFormatError, match="surfaceOp marker"):
        decoder.decode_row(
            {
                "type": "user/message",
                "seq": 0,
                "time": 100,
                "data": {
                    "role": "user",
                    "id": "m",
                    "content": [],
                    "source": {"kind": "user"},
                },
            },
            out,
        )


def test_v3_decoder_recoverable_issue_and_finish() -> None:
    decoder = _V3Codec().create_decoder(_header(3, "s", isSeeded=True), "recoverable")
    out = _Collector()
    decoder.decode_row(
        {
            "type": "user/message",
            "seq": 0,
            "time": 100,
            "data": {
                "role": "user",
                "id": "m",
                "content": [],
                "source": {"kind": "user"},
            },
        },
        out,
    )
    assert out.values == []
    with pytest.raises(SessionFormatError, match="surfaceOp marker"):
        decoder.decode_row(
            {
                "type": "turn/end",
                "seq": 1,
                "time": 101,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
            out,
        )


def test_v4_decoder_version_and_admission() -> None:
    with pytest.raises(SessionFormatError, match="expected format v4"):
        _V4Codec().create_decoder(_header(3, "s"), "strict")
    decoder = _V4Codec().create_decoder(_header(4, "s"), "strict")
    out = _Collector()
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="retired event type"
    ):
        decoder.decode_row(
            {"type": "tool/code-dispatch", "seq": 0, "time": 100, "data": {}},
            out,
        )
    decoder.decode_row(
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}},
        out,
    )
    assert len(out.values) == 1
    assert decoder.finish() == 0


def test_decoder_base_finish_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="decoder must implement finish"):
        _Decoder().finish()


# ---------------------------------------------------------------------------
# v4/v3 admission unit tests
# ---------------------------------------------------------------------------


def _developer_row(**overrides: object) -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": "developer/message",
        "seq": 0,
        "time": 100,
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "role": "developer",
                "id": "d1",
                "content": [{"type": "text", "text": "be nice"}],
                "source": {"kind": "user"},
            },
        },
    }
    row.update(overrides)
    return row


def _system_row(**overrides: object) -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": "system/message",
        "seq": 0,
        "time": 100,
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "sys1",
                "role": "system",
                "content": [{"type": "text", "text": "hello"}],
                "source": {"kind": "system-prompt"},
            },
        },
    }
    row.update(overrides)
    return row


def _tool_result_row(**overrides: object) -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": "tool/result",
        "seq": 0,
        "time": 100,
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "m1",
                "role": "tool",
                "toolCallId": "c1",
                "source": {"kind": "tool", "callId": "c1"},
                "content": [{"type": "text", "text": "3"}],
            },
        },
    }
    row.update(overrides)
    return row


def test_v4_developer_data_valid_and_header_seq() -> None:
    assert_v4_developer_data(_developer_row())
    row = _developer_row()
    row["data"]["message"]["content"] = [
        {"type": "tool-addition", "toolName": "search"},
    ]
    with pytest.raises(SessionFormatError, match="headerSeq"):
        assert_v4_developer_data(row)
    row["data"]["headerSeq"] = 0
    assert_v4_developer_data(row)
    row = _developer_row()
    row["data"]["headerSeq"] = 0
    with pytest.raises(SessionFormatError, match="must omit headerSeq"):
        assert_v4_developer_data(row)


def test_v4_developer_data_errors() -> None:
    with pytest.raises(SessionFormatError, match="data must be an object"):
        assert_v4_developer_data(
            {"type": "developer/message", "seq": 0, "time": 1, "data": "x"}
        )
    row = _developer_row()
    row["data"]["message"]["role"] = "user"
    with pytest.raises(SessionFormatError, match="developer message"):
        assert_v4_developer_data(row)
    row = _developer_row()
    row["data"]["turn"] = 0
    with pytest.raises(SessionFormatError, match="must be positive"):
        assert_v4_developer_data(row)
    row = _developer_row()
    row["data"]["step"] = 0
    with pytest.raises(SessionFormatError, match="must be positive"):
        assert_v4_developer_data(row)
    row = _developer_row()
    row["data"]["message"]["id"] = ""
    with pytest.raises(SessionFormatError, match="requires id, role, content"):
        assert_v4_developer_data(row)
    row = _developer_row()
    row["data"]["message"]["source"] = {"kind": "plugin"}
    with pytest.raises(SessionFormatError, match="requires id, role, content"):
        assert_v4_developer_data(row)
    row = _developer_row()
    row["data"]["message"]["content"] = [
        {"type": "tool-addition", "toolName": "x", "tool": {}}
    ]
    with pytest.raises(SessionFormatError, match="must omit inline tool definitions"):
        assert_v4_developer_data(row)
    row = _developer_row()
    row["data"]["message"]["content"] = [{"type": "tool-addition", "toolName": ""}]
    with pytest.raises(SessionFormatError, match="nonempty toolName"):
        assert_v4_developer_data(row)


def test_v4_developer_data_other_roles() -> None:
    with pytest.raises(SessionFormatError, match="must occur together"):
        assert_v4_developer_data(
            {
                "type": "user/message",
                "seq": 0,
                "time": 1,
                "data": {
                    "role": "developer",
                    "id": "m",
                    "content": [],
                    "source": {"kind": "user"},
                },
            }
        )
    with pytest.raises(SessionFormatError, match="must occur together"):
        assert_v4_developer_data(
            {
                "type": "system/message",
                "seq": 0,
                "time": 1,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "message": {
                        "role": "developer",
                        "id": "m",
                        "content": [],
                        "source": {"kind": "user"},
                    },
                },
            }
        )
    with pytest.raises(SessionFormatError, match="must occur together"):
        assert_v4_developer_data(
            {
                "type": "agent/inbox/spliced",
                "seq": 0,
                "time": 1,
                "data": {"inserted": [{"role": "developer", "id": "m", "content": []}]},
            }
        )
    with pytest.raises(SessionFormatError, match="requires developer role"):
        assert_v4_developer_data(
            {
                "type": "user/message",
                "seq": 0,
                "time": 1,
                "data": {
                    "role": "user",
                    "id": "m",
                    "content": [{"type": "tool-addition", "toolName": "x"}],
                    "source": {"kind": "user"},
                },
            }
        )
    with pytest.raises(SessionFormatError, match="requires developer role"):
        assert_v4_developer_data(
            {
                "type": "compaction/summary",
                "seq": 0,
                "time": 1,
                "data": {"summary": [{"type": "tool-addition", "toolName": "x"}]},
            }
        )
    with pytest.raises(
        SessionFormatError, match="tool-change blocks require developer role"
    ):
        assert_v4_developer_data(
            {
                "type": "assistant/message",
                "seq": 0,
                "time": 1,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "message": {
                        "role": "assistant",
                        "id": "m",
                        "content": [],
                        "source": {"kind": "user"},
                    },
                    "stream": [
                        {
                            "type": "chunk",
                            "chunk": {
                                "type": "block-start",
                                "blockType": "tool-addition",
                            },
                        },
                    ],
                },
            }
        )
    with pytest.raises(SessionFormatError, match="requires developer role"):
        assert_v4_developer_data(
            {
                "type": "assistant/message",
                "seq": 0,
                "time": 1,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "message": {
                        "role": "assistant",
                        "id": "m",
                        "content": [],
                        "source": {"kind": "user"},
                    },
                    "stream": [
                        {
                            "type": "chunk",
                            "chunk": {
                                "type": "block-end",
                                "block": {"type": "tool-addition", "toolName": "x"},
                            },
                        },
                    ],
                },
            }
        )
    with pytest.raises(SessionFormatError, match="deferLoading must be true"):
        assert_v4_developer_data(
            {
                "type": "request/header",
                "seq": 0,
                "time": 1,
                "data": {"header": {"tools": [{"name": "x", "deferLoading": False}]}},
            }
        )
    row = {
        "type": "request/header",
        "seq": 0,
        "time": 1,
        "data": {
            "header": {"tools": [{"name": "x", "deferLoading": True}, {"name": "y"}]}
        },
    }
    assert_v4_developer_data(row)
    assert_v4_developer_data({"type": "turn/start", "seq": 0, "time": 1, "data": "x"})


def test_assert_tool_change_and_map_event_messages() -> None:
    assert _assert_tool_change("not an object", False) is None
    assert _assert_tool_change({"type": "text", "text": "x"}, False) is None
    with pytest.raises(SessionFormatError, match="requires developer role"):
        _assert_tool_change({"type": "tool-removal", "toolName": "x"}, False)
    assert_v4_retired_syntax("not an object")
    assert_v4_developer_data(
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}
    )
    event = {
        "type": "user/message",
        "seq": 0,
        "time": 1,
        "data": {"role": "user", "id": "m", "content": []},
    }
    assert _map_event_messages(event, lambda message: message) is event
    assert (
        _map_event_messages(
            {"type": "turn/start", "seq": 0, "time": 1, "data": "x"},
            lambda message: message,
        )["data"]
        == "x"
    )
    with pytest.raises(SessionFormatError, match="requires a message"):
        _map_event_messages(
            {"type": "assistant/message", "seq": 0, "time": 1, "data": {"turn": 1}},
            lambda message: message,
        )
    with pytest.raises(SessionFormatError, match="requires message array"):
        _map_event_messages(
            {
                "type": "agent/inbox/spliced",
                "seq": 0,
                "time": 1,
                "data": {"inserted": "x"},
            },
            lambda message: message,
        )
    with pytest.raises(SessionFormatError, match="requires message objects"):
        _map_event_messages(
            {
                "type": "agent/inbox/spliced",
                "seq": 0,
                "time": 1,
                "data": {"inserted": ["x"]},
            },
            lambda message: message,
        )
    mapped = _map_event_messages(
        {
            "type": "session/title-llm-request",
            "seq": 0,
            "time": 1,
            "data": {"messages": [{"role": "user", "id": "m"}]},
        },
        lambda message: {**message, "id": "n"},
    )
    assert mapped["data"]["messages"][0]["id"] == "n"


def test_v4_retired_syntax() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="retired event type"
    ):
        assert_v4_retired_syntax(
            {"type": "tool/code-dispatch", "seq": 0, "time": 1, "data": {}}
        )
    assert_v4_retired_syntax(
        {
            "type": "tool/code-dispatch",
            "seq": 0,
            "time": 1,
            "data": {},
            "ignorable": True,
        }
    )
    with pytest.raises(SessionFormatError, match="requires data and header objects"):
        assert_v4_retired_syntax(
            {"type": "request/header", "seq": 0, "time": 1, "data": {}}
        )
    with pytest.raises(SessionFormatError, match="retired header.system"):
        assert_v4_retired_syntax(
            {
                "type": "request/header",
                "seq": 0,
                "time": 1,
                "data": {"header": {"system": "x"}},
            }
        )
    assert_v4_retired_syntax(
        {"type": "request/header", "seq": 0, "time": 1, "data": {"header": {}}}
    )
    bad = {"type": "tool-result", "text": "x"}
    with pytest.raises(
        SessionFormatError, match="must not contain a released tool-result wrapper"
    ):
        assert_v4_retired_syntax(
            {"type": "user/message", "seq": 0, "time": 1, "data": {"content": [bad]}}
        )
    with pytest.raises(
        SessionFormatError, match="must not contain a released tool-result wrapper"
    ):
        assert_v4_retired_syntax(
            {
                "type": "assistant/message",
                "seq": 0,
                "time": 1,
                "data": {"message": {"content": [bad]}},
            }
        )
    with pytest.raises(
        SessionFormatError, match="must not contain a released tool-result wrapper"
    ):
        assert_v4_retired_syntax(
            {
                "type": "agent/inbox/spliced",
                "seq": 0,
                "time": 1,
                "data": {"inserted": [{"content": [bad]}]},
            }
        )
    with pytest.raises(
        SessionFormatError, match="must not contain a released tool-result wrapper"
    ):
        assert_v4_retired_syntax(
            {
                "type": "compaction/summary",
                "seq": 0,
                "time": 1,
                "data": {"rawOutput": [bad]},
            }
        )
    with pytest.raises(
        SessionFormatError, match="must not contain a released tool-result wrapper"
    ):
        assert_v4_retired_syntax(
            {
                "type": "tool/ptc-dispatch",
                "seq": 0,
                "time": 1,
                "data": {"content": [bad]},
            }
        )
    with pytest.raises(
        SessionFormatError, match="must not contain a released tool-result wrapper"
    ):
        assert_v4_retired_syntax(
            {
                "type": "assistant/message",
                "seq": 0,
                "time": 1,
                "data": {
                    "stream": [
                        {"type": "chunk", "chunk": {"type": "block-end", "block": bad}}
                    ]
                },
            }
        )
    with pytest.raises(
        SessionFormatError, match="must not contain a released tool-result wrapper"
    ):
        assert_v4_retired_syntax(
            {
                "type": "assistant/message",
                "seq": 0,
                "time": 1,
                "data": {
                    "stream": [
                        {
                            "type": "chunk",
                            "chunk": {
                                "type": "block-start",
                                "blockType": "tool-result",
                            },
                        }
                    ]
                },
            }
        )
    assert_v4_retired_syntax(
        {
            "type": "assistant/message",
            "seq": 0,
            "time": 1,
            "data": {
                "stream": [
                    "not an object",
                    {"type": "note"},
                    {"type": "chunk", "chunk": "x"},
                ]
            },
        }
    )


def test_v4_system_message_fields() -> None:
    assert_v4_system_message_fields(_system_row())
    assert_v4_system_message_fields(
        {"type": "turn/start", "seq": 0, "time": 1, "data": {}}
    )
    with pytest.raises(SessionFormatError, match="system/message data"):
        assert_v4_system_message_fields(
            {"type": "system/message", "seq": 0, "time": 1, "data": "x"}
        )
    row = _system_row()
    row["data"]["turn"] = 0
    with pytest.raises(SessionFormatError, match="must be positive"):
        assert_v4_system_message_fields(row)
    row = _system_row()
    row["data"]["message"] = "x"
    with pytest.raises(SessionFormatError, match="system message"):
        assert_v4_system_message_fields(row)
    row = _system_row()
    row["data"]["message"]["id"] = ""
    with pytest.raises(SessionFormatError, match="requires a string id"):
        assert_v4_system_message_fields(row)
    row = _system_row()
    row["data"]["message"]["role"] = "user"
    with pytest.raises(SessionFormatError, match="requires system role"):
        assert_v4_system_message_fields(row)
    row = _system_row()
    row["data"]["message"]["content"] = "x"
    with pytest.raises(SessionFormatError, match="requires content array"):
        assert_v4_system_message_fields(row)
    row = _system_row()
    row["data"]["message"]["content"] = ["x"]
    with pytest.raises(SessionFormatError, match="system content block"):
        assert_v4_system_message_fields(row)
    row = _system_row()
    row["data"]["message"]["content"] = [{"type": ""}]
    with pytest.raises(SessionFormatError, match="requires a string type"):
        assert_v4_system_message_fields(row)
    row = _system_row()
    row["data"]["message"]["content"] = [{"type": "text", "text": 3}]
    with pytest.raises(SessionFormatError, match="text requires a string"):
        assert_v4_system_message_fields(row)
    row = _system_row()
    row["data"]["message"]["content"] = [
        {"type": "tool-call", "id": "c", "name": "", "arguments": "{}"}
    ]
    with pytest.raises(SessionFormatError, match="tool-call name requires a string"):
        assert_v4_system_message_fields(row)
    row = _system_row()
    row["data"]["message"]["content"] = [
        {"type": "tool-call", "id": "c", "name": "f", "arguments": 3}
    ]
    with pytest.raises(SessionFormatError, match="arguments requires a string"):
        assert_v4_system_message_fields(row)
    row = _system_row()
    row["data"]["message"]["content"] = [{"type": "tool-result", "text": "x"}]
    with pytest.raises(
        SessionFormatError, match="rejects retired tool-result wrappers"
    ):
        assert_v4_system_message_fields(row)


def test_v4_system_message_image_fields() -> None:
    def image_block(**overrides: object) -> dict[str, Any]:
        attachment: dict[str, Any] = {
            "attachmentId": "a1",
            "mediaType": "image/png",
            "bytes": 10,
            "width": 4,
            "height": 4,
        }
        attachment.update(overrides)
        return {"type": "image", "attachment": attachment}

    row = _system_row()
    row["data"]["message"]["content"] = [image_block()]
    assert_v4_system_message_fields(row)
    row = _system_row()
    row["data"]["message"]["content"] = [{"type": "image", "attachment": "x"}]
    with pytest.raises(SessionFormatError, match="image attachment"):
        assert_v4_system_message_fields(row)
    row = _system_row()
    row["data"]["message"]["content"] = [image_block(attachmentId="")]
    with pytest.raises(SessionFormatError, match="attachmentId requires a string"):
        assert_v4_system_message_fields(row)
    row = _system_row()
    row["data"]["message"]["content"] = [image_block(mediaType="image/tiff")]
    with pytest.raises(SessionFormatError, match="supported mediaType"):
        assert_v4_system_message_fields(row)
    row = _system_row()
    row["data"]["message"]["content"] = [image_block(bytes="x")]
    with pytest.raises(SessionFormatError, match="image bytes"):
        assert_v4_system_message_fields(row)
    row = _system_row()
    row["data"]["message"]["content"] = [image_block(width=0)]
    with pytest.raises(SessionFormatError, match="image width must be positive"):
        assert_v4_system_message_fields(row)
    row = _system_row()
    row["data"]["message"]["content"] = [image_block(name=3)]
    with pytest.raises(SessionFormatError, match="image name requires a string"):
        assert_v4_system_message_fields(row)
    row = _system_row()
    row["data"]["message"]["content"] = [
        image_block(originalDimensions={"width": 0, "height": 4})
    ]
    with pytest.raises(SessionFormatError, match="original width must be positive"):
        assert_v4_system_message_fields(row)


def test_v4_tool_result_message() -> None:
    assert_v4_tool_result_message(_tool_result_row())
    assert_v4_tool_result_message(
        {"type": "turn/start", "seq": 0, "time": 1, "data": {}}
    )
    with pytest.raises(SessionFormatError, match="tool/result.*data"):
        assert_v4_tool_result_message(
            {"type": "tool/result", "seq": 0, "time": 1, "data": "x"}
        )
    row = _tool_result_row()
    row["data"]["message"] = "x"
    with pytest.raises(SessionFormatError, match="tool/result.*message"):
        assert_v4_tool_result_message(row)
    row = _tool_result_row()
    row["data"]["message"]["id"] = ""
    with pytest.raises(SessionFormatError, match="requires a first-class message"):
        assert_v4_tool_result_message(row)
    row = _tool_result_row()
    row["data"]["message"]["role"] = "user"
    with pytest.raises(SessionFormatError, match="requires a tool-role message"):
        assert_v4_tool_result_message(row)
    row = _tool_result_row()
    row["data"]["message"]["toolCallId"] = "c2"
    with pytest.raises(SessionFormatError, match="requires toolCallId matching"):
        assert_v4_tool_result_message(row)
    row = _tool_result_row()
    row["data"]["message"]["source"] = {"kind": "user", "callId": "c1"}
    with pytest.raises(SessionFormatError, match="requires a tool source"):
        assert_v4_tool_result_message(row)
    row = _tool_result_row()
    row["data"]["message"]["content"] = "x"
    with pytest.raises(SessionFormatError, match="requires array content"):
        assert_v4_tool_result_message(row)
    row = _tool_result_row()
    row["data"]["message"]["content"] = [{"type": "tool-result", "text": "x"}]
    with pytest.raises(
        SessionFormatError, match="must not contain a released tool-result wrapper"
    ):
        assert_v4_tool_result_message(row)
    row = _tool_result_row()
    row["data"]["message"]["isError"] = "yes"
    with pytest.raises(SessionFormatError, match="isError must be boolean"):
        assert_v4_tool_result_message(row)
    row = _tool_result_row()
    row["data"]["error"] = {"code": "E"}
    with pytest.raises(SessionFormatError, match="carries error metadata"):
        assert_v4_tool_result_message(row)


def _fork_result_row(**overrides: object) -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": "tool/result",
        "seq": 0,
        "time": 100,
        "surfaceOp": "append",
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "forked-tool-result-c1-0",
                "role": "tool",
                "toolCallId": "c1",
                "source": {"kind": "tool", "callId": "c1"},
                "isError": True,
                "content": [{"type": "text", "text": "not started"}],
            },
            "error": {"code": "TOOL_NOT_STARTED", "name": "ToolNotStartedError"},
        },
    }
    row.update(overrides)
    return row


def test_v4_fork_result() -> None:
    assert_v4_fork_result(_fork_result_row())
    assert_v4_fork_result({"type": "turn/start", "seq": 0, "time": 1, "data": {}})
    assert_v4_fork_result({"type": "tool/result", "seq": 0, "time": 1, "data": "x"})
    row = _fork_result_row()
    row["data"]["error"] = {"code": "OTHER"}
    assert_v4_fork_result(row)
    row = _fork_result_row()
    row["data"]["message"]["id"] = "plain-id"
    assert_v4_fork_result(row)
    row = _fork_result_row()
    row["data"]["error"]["name"] = "OtherError"
    with pytest.raises(SessionFormatError, match="invalid V4 not-started fork result"):
        assert_v4_fork_result(row)
    row = _fork_result_row()
    row["data"]["message"]["id"] = "forked-tool-result-c1-01"
    with pytest.raises(SessionFormatError, match="invalid V4 not-started fork result"):
        assert_v4_fork_result(row)
    row = _fork_result_row()
    row["data"]["message"]["id"] = "forked-tool-result-c1-x"
    with pytest.raises(SessionFormatError, match="invalid V4 not-started fork result"):
        assert_v4_fork_result(row)
    row = _fork_result_row()
    row["data"]["message"]["id"] = "forked-tool-result-c1-2"
    row["seq"] = 5
    row["surfaceOp"] = {"op": "replace"}
    row["sourceEventSeqs"] = [2]
    assert_v4_fork_result(row)
    row = _fork_result_row()
    row["data"]["message"]["id"] = "forked-tool-result-c1-2"
    row["seq"] = 5
    row["surfaceOp"] = {"op": "replace"}
    with pytest.raises(SessionFormatError, match="invalid V4 not-started fork result"):
        assert_v4_fork_result(row)


def test_v4_source_row_admission_and_message_sources() -> None:
    with pytest.raises(SessionFormatError, match="producer-owned source kind"):
        assert_v4_source_row_admission(
            {
                "type": "user/message",
                "seq": 0,
                "time": 1,
                "data": {"source": {"kind": "plugin"}},
            }
        )
    with pytest.raises(SessionFormatError, match="producer-owned source kind"):
        assert_v4_source_row_admission(
            {
                "type": "tool/result",
                "seq": 0,
                "time": 1,
                "data": {"message": {"source": {"kind": "plugin"}}},
            }
        )
    with pytest.raises(SessionFormatError, match="producer-owned source kind"):
        assert_v4_source_row_admission(
            {
                "type": "agent/inbox/spliced",
                "seq": 0,
                "time": 1,
                "data": {"inserted": [{"source": {"kind": "plugin"}}]},
            }
        )
    with pytest.raises(SessionFormatError, match="producer-owned source kind"):
        assert_v4_source_row_admission(
            {
                "type": "session/title-llm-request",
                "seq": 0,
                "time": 1,
                "data": {"messages": [{"source": {"kind": "plugin"}}]},
            }
        )
    assert_v4_source_row_admission("not an object")
    assert_v4_source_row_admission(
        {"type": "agent/inbox/spliced", "seq": 0, "time": 1, "data": {"inserted": "x"}}
    )
    assert_v4_source_row_admission(
        {
            "type": "agent/inbox/spliced",
            "seq": 0,
            "time": 1,
            "data": {"inserted": ["x", {"source": {"kind": "user"}}]},
        }
    )
    assert_v4_source_row_admission(
        {"type": "turn/start", "seq": 0, "time": 1, "data": {}}
    )
    with pytest.raises(SessionFormatError, match="producer-owned source kind"):
        assert_v4_message_sources(
            {
                "type": "user/message",
                "seq": 0,
                "time": 1,
                "data": {"source": {"kind": "plugin"}},
            }
        )
    assert_v4_message_sources(
        {
            "type": "user/message",
            "seq": 0,
            "time": 1,
            "data": {"source": {"kind": "user"}},
        }
    )


def test_v4_row_admission() -> None:
    row = _developer_row()
    row["ignorable"] = True
    assert_v4_row_admission(row)
    row["data"]["message"]["role"] = "user"
    with pytest.raises(SessionFormatError, match="developer/message"):
        assert_v4_row_admission(row, KNOWN_SESSION_EVENT_TYPES)
    assert_v4_row_admission(_tool_result_row())
    with pytest.raises(SessionFormatError, match="requires a first-class message"):
        bad = _tool_result_row()
        bad["data"]["message"]["id"] = ""
        assert_v4_row_admission(bad)
    assert_v4_row_admission(
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}
    )


def test_v3_event_admission_and_structural() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="unknown event type"
    ):
        assert_v3_event_admission(
            {"type": "tool/code-dispatch", "seq": 0, "time": 1, "data": {}}
        )
    assert_v3_event_admission(
        {
            "type": "tool/code-dispatch",
            "seq": 0,
            "time": 1,
            "data": {},
            "ignorable": True,
        }
    )
    assert_v3_structural_row("not an object")
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="retired header.system"
    ):
        assert_v3_structural_row(
            {
                "type": "request/header",
                "seq": 0,
                "time": 1,
                "data": {"header": {"system": "x"}},
            }
        )
    with pytest.raises(SessionFormatError, match="request/header data"):
        assert_v3_structural_row(
            {"type": "request/header", "seq": 0, "time": 1, "data": "x"}
        )
    valid_system = {
        "turn": 1,
        "step": 1,
        "message": {
            "id": "s",
            "role": "system",
            "source": {"kind": "plugin", "plugin": "system-prompt"},
            "content": [],
        },
    }
    assert_v3_structural_row(
        {"type": "system/message", "seq": 0, "time": 1, "data": valid_system}
    )
    with pytest.raises(SessionFormatError, match="requires plugin source"):
        bad = {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "s",
                "role": "system",
                "source": {"kind": "user"},
                "content": [],
            },
        }
        assert_v3_structural_row(
            {"type": "system/message", "seq": 0, "time": 1, "data": bad}
        )
    with pytest.raises(SessionFormatError, match="turn.*must be positive"):
        bad = {
            "turn": 0,
            "step": 1,
            "message": {
                "id": "s",
                "role": "system",
                "source": {"kind": "plugin", "plugin": "p"},
                "content": [],
            },
        }
        assert_v3_structural_row(
            {"type": "system/message", "seq": 0, "time": 1, "data": bad}
        )
    with pytest.raises(SessionFormatError, match="requires an id and system role"):
        bad = {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "",
                "role": "system",
                "source": {"kind": "plugin", "plugin": "p"},
                "content": [],
            },
        }
        assert_v3_structural_row(
            {"type": "system/message", "seq": 0, "time": 1, "data": bad}
        )
    with pytest.raises(SessionFormatError, match="has unexpected member"):
        bad = {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "s",
                "role": "system",
                "source": {"kind": "plugin", "plugin": "p"},
                "content": [],
            },
            "extra": 1,
        }
        assert_v3_structural_row(
            {"type": "system/message", "seq": 0, "time": 1, "data": bad}
        )


def test_v3_event_envelope_branches() -> None:
    base = {
        "type": "user/message",
        "seq": 0,
        "time": 100,
        "data": {"role": "user", "id": "m", "content": [], "source": {"kind": "user"}},
        "surfaceOp": "append",
    }
    assert_v3_event(base)
    with pytest.raises(SessionFormatError, match="has unexpected member"):
        assert_v3_event({**base, "extra": 1})
    with pytest.raises(SessionFormatError, match="type must be a string"):
        assert_v3_event({**base, "type": 3})
    with pytest.raises(SessionFormatError, match="ignorable must be true"):
        assert_v3_event({**base, "ignorable": False})
    with pytest.raises(SessionFormatError, match="requires a surfaceOp marker"):
        assert_v3_event({k: v for k, v in base.items() if k != "surfaceOp"})
    with pytest.raises(SessionFormatError, match="requires exact replace fields"):
        assert_v3_event(
            {**base, "seq": 5, "surfaceOp": {"op": "replace", "startSeq": 1}}
        )
    with pytest.raises(
        SessionFormatError, match="replacement endpoints must reference earlier"
    ):
        assert_v3_event(
            {
                **base,
                "seq": 5,
                "surfaceOp": {"op": "replace", "startSeq": 1, "endSeq": 6},
            }
        )
    with pytest.raises(SessionFormatError, match="embeds its stream"):
        assert_v3_event(
            {
                "type": "assistant/message",
                "seq": 5,
                "time": 100,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "message": {
                        "id": "m",
                        "role": "assistant",
                        "content": [],
                        "source": {"kind": "user"},
                    },
                    "stream": [],
                },
                "surfaceOp": "append",
                "sourceEventSeqs": [1],
            }
        )
    with pytest.raises(
        SessionFormatError, match="sourceEventSeqs must be a non-empty array"
    ):
        assert_v3_event({**base, "sourceEventSeqs": []})
    with pytest.raises(
        SessionFormatError, match="sourceEventSeqs must be unique earlier seqs"
    ):
        assert_v3_event({**base, "seq": 3, "sourceEventSeqs": [1, 1]})
    with pytest.raises(
        SessionFormatError, match="empty optional header fields must be omitted"
    ):
        assert_v3_event(
            {
                "type": "request/header",
                "seq": 0,
                "time": 100,
                "data": {"header": {"tools": []}},
            }
        )
    with pytest.raises(
        SessionFormatError, match="empty optional header fields must be omitted"
    ):
        assert_v3_event(
            {
                "type": "request/header",
                "seq": 0,
                "time": 100,
                "data": {"header": {"adapterDefaults": {}}},
            }
        )
    with pytest.raises(SessionFormatError, match="carries error metadata"):
        assert_v3_event(
            {
                "type": "tool/result",
                "seq": 0,
                "time": 100,
                "surfaceOp": "append",
                "data": {
                    "turn": 1,
                    "step": 1,
                    "message": {
                        "id": "m",
                        "role": "tool",
                        "content": [{"type": "text", "text": "x"}],
                    },
                    "error": {"code": "E"},
                },
            }
        )
    assert_v3_event(
        {
            "type": "tool/result",
            "seq": 0,
            "time": 100,
            "surfaceOp": "append",
            "data": {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "m",
                    "role": "tool",
                    "content": [{"type": "tool-result", "text": "x", "isError": True}],
                },
                "error": {"code": "E"},
            },
        }
    )
    # feedback/message-put is known via the feedback branch, not a surface
    # type, so no surfaceOp marker is required for it — and one is refused.
    assert_v3_event({"type": "feedback/message-put", "seq": 0, "time": 100, "data": {}})
    with pytest.raises(SessionFormatError, match="has unexpected member"):
        assert_v3_event(
            {
                "type": "feedback/message-put",
                "seq": 0,
                "time": 100,
                "data": {},
                "surfaceOp": "append",
            },
            RELEASED_V3_EVENT_TYPES,
        )


# ---------------------------------------------------------------------------
# Restore validation
# ---------------------------------------------------------------------------


def _artifact(
    version: int,
    events: list[dict[str, Any]],
    *,
    seeded: bool = False,
    inherited: int = 0,
    session_id: str = "a",
) -> SessionFormatArtifact:
    header = {
        "version": version,
        "id": session_id,
        "createdAt": 1,
        "isSeeded": seeded,
        "delegationDepth": 0,
    }
    return SessionFormatArtifact(
        header=header, inherited_event_count=inherited, events=events
    )


def _turn_pair() -> list[dict[str, Any]]:
    return [
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}},
        {
            "type": "turn/end",
            "seq": 1,
            "time": 101,
            "data": {"turn": 1, "reason": {"kind": "done"}},
        },
    ]


def test_restore_v4_valid_and_density() -> None:
    assert (
        restore_released_v4_artifact(
            _artifact(4, _turn_pair()), KNOWN_SESSION_EVENT_TYPES
        ).header["version"]
        == 4
    )
    with pytest.raises(SessionFormatError, match="exceeds its events"):
        restore_released_v4_artifact(
            _artifact(4, _turn_pair(), inherited=5), KNOWN_SESSION_EVENT_TYPES
        )
    with pytest.raises(SessionFormatError, match="unseeded format v4"):
        restore_released_v4_artifact(
            _artifact(4, _turn_pair(), inherited=1), KNOWN_SESSION_EVENT_TYPES
        )
    events = _turn_pair()
    events[1]["seq"] = 5
    with pytest.raises(SessionFormatError, match="is not dense"):
        restore_released_v4_artifact(_artifact(4, events), KNOWN_SESSION_EVENT_TYPES)


def test_restore_v4_unknown_and_markers() -> None:
    events = _turn_pair()
    events.insert(
        1,
        {"type": "vendor/widget", "seq": 1, "time": 100, "data": {}, "ignorable": True},
    )
    events[2]["seq"] = 2
    assert (
        restore_released_v4_artifact(_artifact(4, events), KNOWN_SESSION_EVENT_TYPES)
        is not None
    )
    events = _turn_pair()
    events.insert(1, {"type": "vendor/widget", "seq": 1, "time": 100, "data": {}})
    events[2]["seq"] = 2
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="unknown event type"
    ):
        restore_released_v4_artifact(_artifact(4, events), KNOWN_SESSION_EVENT_TYPES)
    events = [
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}},
        {
            "type": "session/end-seed",
            "seq": 1,
            "time": 101,
            "data": {"inherited": True},
        },
        {
            "type": "turn/end",
            "seq": 2,
            "time": 102,
            "data": {"turn": 1, "reason": {"kind": "done"}},
        },
    ]
    assert (
        restore_released_v4_artifact(
            _artifact(4, events, seeded=True, inherited=1), KNOWN_SESSION_EVENT_TYPES
        )
        is not None
    )
    with pytest.raises(SessionFormatError, match="seeded header disagrees"):
        restore_released_v4_artifact(
            _artifact(4, events, seeded=True, inherited=2), KNOWN_SESSION_EVENT_TYPES
        )
    with pytest.raises(
        SessionFormatError, match="unseeded Session contains an inherited"
    ):
        restore_released_v4_artifact(_artifact(4, events), KNOWN_SESSION_EVENT_TYPES)


def test_restore_v4_admission_failures() -> None:
    events = [
        {
            "type": "request/header",
            "seq": 0,
            "time": 100,
            "data": {"header": {"system": "x"}},
        },
    ]
    with pytest.raises(SessionFormatError, match="retired header.system"):
        restore_released_v4_artifact(_artifact(4, events), KNOWN_SESSION_EVENT_TYPES)
    bad_system = _system_row()
    bad_system["data"]["message"]["role"] = "user"
    with pytest.raises(SessionFormatError, match="requires system role"):
        restore_released_v4_artifact(
            _artifact(4, [bad_system]), KNOWN_SESSION_EVENT_TYPES
        )
    bad_tool = _tool_result_row()
    bad_tool["data"]["message"]["role"] = "user"
    with pytest.raises(SessionFormatError, match="requires a tool-role message"):
        restore_released_v4_artifact(
            _artifact(4, [bad_tool]), KNOWN_SESSION_EVENT_TYPES
        )
    bad_fork = _fork_result_row()
    bad_fork["data"]["error"]["name"] = "OtherError"
    with pytest.raises(SessionFormatError, match="invalid V4 not-started fork result"):
        restore_released_v4_artifact(
            _artifact(4, [bad_fork]), KNOWN_SESSION_EVENT_TYPES
        )


def test_restore_v4_relationships() -> None:
    events = _turn_pair()
    events.append(
        {
            "type": "session-log-deepseek/delivery-accepted",
            "seq": 2,
            "time": 102,
            "data": {"sessionId": "other", "throughSeq": 1, "sessionFormatVersion": 4},
        }
    )
    with pytest.raises(SessionFormatError, match="names the wrong Session"):
        restore_released_v4_artifact(_artifact(4, events), KNOWN_SESSION_EVENT_TYPES)
    catalog = {
        "version": 0,
        "childId": "c1",
        "childCreatedAt": 500,
        "mode": "one-shot",
        "label": "L",
    }
    events = [
        {"type": "subagent/catalog", "seq": 0, "time": 100, "data": catalog},
        {"type": "subagent/catalog", "seq": 1, "time": 101, "data": catalog},
    ]
    with pytest.raises(SessionFormatError, match="duplicate catalog child"):
        restore_released_v4_artifact(_artifact(4, events), KNOWN_SESSION_EVENT_TYPES)
    events = [
        {"type": "subagent/catalog", "seq": 0, "time": 100, "data": catalog},
        {"type": "turn/start", "seq": 1, "time": 101, "data": {"turn": 1}},
    ]
    assert_released_v4_relationships(_artifact(4, events), KNOWN_SESSION_EVENT_TYPES)
    events = [
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}},
        {
            "type": "session-log-deepseek/delivery-accepted",
            "seq": 1,
            "time": 101,
            "data": {"sessionId": "a", "throughSeq": 0, "sessionFormatVersion": 4},
        },
    ]
    assert_released_v4_relationships(_artifact(4, events), KNOWN_SESSION_EVENT_TYPES)


def test_restore_v2_coordinates() -> None:
    assert _restore_released_v2_artifact(_artifact(2, _turn_pair()), None) is not None
    with pytest.raises(SessionFormatError, match="exceeds its events"):
        _restore_released_v2_artifact(_artifact(2, _turn_pair(), inherited=5), None)
    with pytest.raises(SessionFormatError, match="unseeded format v2"):
        _restore_released_v2_artifact(_artifact(2, _turn_pair(), inherited=1), None)
    with pytest.raises(SessionFormatError, match="must be a JSON object"):
        _restore_released_v2_artifact(_artifact(2, ["not an object"]), None)
    with pytest.raises(SessionFormatError, match="type must be a string"):
        _restore_released_v2_artifact(
            _artifact(2, [{"type": 3, "seq": 0, "time": 100, "data": {}}]), None
        )
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="unknown event type"
    ):
        _restore_released_v2_artifact(
            _artifact(
                2, [{"type": "vendor/widget", "seq": 0, "time": 100, "data": {}}]
            ),
            None,
        )
    assert (
        _restore_released_v2_artifact(
            _artifact(
                2,
                [
                    {
                        "type": "vendor/widget",
                        "seq": 0,
                        "time": 100,
                        "data": {},
                        "ignorable": True,
                    }
                ],
            ),
            None,
        )
        is not None
    )
    assert (
        _restore_released_v2_artifact(
            _artifact(
                2, [{"type": "developer/message", "seq": 0, "time": 100, "data": {}}]
            ),
            KNOWN_SESSION_EVENT_TYPES,
        )
        is not None
    )
    events = _turn_pair()
    events[0]["surfaceOp"] = "append"
    with pytest.raises(SessionFormatError, match="has unexpected member"):
        _restore_released_v2_artifact(_artifact(2, events), None)
    events = _turn_pair()
    events[1]["seq"] = 5
    with pytest.raises(SessionFormatError, match="is not dense"):
        _restore_released_v2_artifact(_artifact(2, events), None)
    events = _turn_pair()
    events[0]["time"] = "x"
    with pytest.raises(SessionFormatError, match="time"):
        _restore_released_v2_artifact(_artifact(2, events), None)
    events = _turn_pair()
    events[0]["ignorable"] = False
    with pytest.raises(SessionFormatError, match="ignorable must be true"):
        _restore_released_v2_artifact(_artifact(2, events), None)
    with pytest.raises(SessionFormatError, match="session/end-seed"):
        _restore_released_v2_artifact(
            _artifact(
                2, [{"type": "session/end-seed", "seq": 0, "time": 100, "data": "x"}]
            ),
            None,
        )


def test_restore_v2_end_seed_agreement() -> None:
    events = [
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}},
        {
            "type": "session/end-seed",
            "seq": 1,
            "time": 101,
            "data": {"inherited": True},
        },
    ]
    assert (
        _restore_released_v2_artifact(
            _artifact(2, events, seeded=True, inherited=1), None
        )
        is not None
    )
    with pytest.raises(SessionFormatError, match="seeded header disagrees"):
        _restore_released_v2_artifact(
            _artifact(2, events, seeded=True, inherited=0), None
        )
    with pytest.raises(
        SessionFormatError, match="unseeded Session contains an inherited"
    ):
        _restore_released_v2_artifact(_artifact(2, events), None)


def _v3_system_event(seq: int, time: int, surface_op: Any) -> dict[str, Any]:
    return {
        "type": "system/message",
        "seq": seq,
        "time": time,
        "surfaceOp": surface_op,
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "sys",
                "role": "system",
                "source": {"kind": "plugin", "plugin": "system-prompt"},
                "content": [],
            },
        },
    }


def _v3_valid_events() -> list[dict[str, Any]]:
    return [
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}},
        {"type": "step/start", "seq": 1, "time": 101, "data": {"turn": 1, "step": 1}},
        _v3_system_event(2, 102, "append"),
        {
            "type": "user/message",
            "seq": 3,
            "time": 103,
            "surfaceOp": "append",
            "data": {
                "role": "user",
                "id": "m",
                "content": [],
                "source": {"kind": "user"},
            },
        },
        {"type": "step/end", "seq": 4, "time": 104, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": 5,
            "time": 105,
            "data": {"turn": 1, "reason": {"kind": "done"}},
        },
    ]


def test_restore_v3_valid_and_head_rules() -> None:
    assert (
        restore_released_v3_artifact(
            _artifact(3, _v3_valid_events()), RELEASED_V3_EVENT_TYPES
        )
        is not None
    )
    events = _v3_valid_events()
    events[2]["data"]["turn"] = 2
    with pytest.raises(SessionFormatError, match="does not match an open step"):
        restore_released_v3_artifact(_artifact(3, events), RELEASED_V3_EVENT_TYPES)
    events = _v3_valid_events()
    del events[1]
    events[2]["seq"] = 2
    with pytest.raises(SessionFormatError, match="does not match an open step"):
        restore_released_v3_artifact(_artifact(3, events), RELEASED_V3_EVENT_TYPES)
    events = _v3_valid_events()
    events[2], events[3] = events[3], events[2]
    events[2]["seq"] = 2
    events[3]["seq"] = 3
    with pytest.raises(SessionFormatError, match="protected first surface head"):
        restore_released_v3_artifact(_artifact(3, events), RELEASED_V3_EVENT_TYPES)
    events = _v3_valid_events()
    events.insert(
        4, _v3_system_event(4, 105, {"op": "replace", "startSeq": 2, "endSeq": 3})
    )
    events[5]["seq"] = 5
    events[6]["seq"] = 6
    with pytest.raises(SessionFormatError, match="must replace exactly the current"):
        restore_released_v3_artifact(_artifact(3, events), RELEASED_V3_EVENT_TYPES)
    events = _v3_valid_events()
    events.insert(
        4, _v3_system_event(4, 105, {"op": "replace", "startSeq": 2, "endSeq": 2})
    )
    events[5]["seq"] = 5
    events[6]["seq"] = 6
    assert (
        restore_released_v3_artifact(_artifact(3, events), RELEASED_V3_EVENT_TYPES)
        is not None
    )


def test_restore_v3_shadow_and_density() -> None:
    events = _v3_valid_events()
    events.append(
        {
            "type": "user/message",
            "seq": 6,
            "time": 106,
            "surfaceOp": {"op": "replace", "startSeq": 2, "endSeq": 3},
            "sourceEventSeqs": [2],
            "data": {
                "role": "user",
                "id": "m2",
                "content": [],
                "source": {"kind": "user"},
            },
        }
    )
    with pytest.raises(
        SessionFormatError, match="cannot shadow the protected system head"
    ):
        restore_released_v3_artifact(_artifact(3, events), RELEASED_V3_EVENT_TYPES)
    events = _v3_valid_events()
    events.append(
        {
            "type": "compaction/summary",
            "seq": 6,
            "time": 106,
            "data": {
                "compactionId": "c1",
                "summary": [],
                "shadowedRange": {"start": 0, "end": 3},
                "shadowedSeqs": [2],
                "shadowedTokenCount": 1,
                "provider": "p",
                "model": "m",
            },
        }
    )
    with pytest.raises(SessionFormatError, match="compaction cannot shadow"):
        restore_released_v3_artifact(_artifact(3, events), RELEASED_V3_EVENT_TYPES)
    events = _v3_valid_events()
    events.append({"type": "tool/code-dispatch", "seq": 6, "time": 106, "data": {}})
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="unknown event type"
    ):
        restore_released_v3_artifact(_artifact(3, events), RELEASED_V3_EVENT_TYPES)
    events = _v3_valid_events()
    events[5]["seq"] = 9
    with pytest.raises(SessionFormatError, match="is not dense"):
        restore_released_v3_artifact(_artifact(3, events), RELEASED_V3_EVENT_TYPES)


# ---------------------------------------------------------------------------
# v2 event assertion internals
# ---------------------------------------------------------------------------


def test_is_repair_identity_variants() -> None:
    assert _is_repair_identity("interrupted-tool-result-c1-0", "c1") is True
    assert _is_repair_identity("interrupted-tool-result-c1-12", "c1") is True
    assert _is_repair_identity("interrupted-tool-result-c1-0", 3) is False
    assert _is_repair_identity(3, "c1") is False
    assert _is_repair_identity("plain-id", "c1") is False
    assert _is_repair_identity("interrupted-tool-result-c1-", "c1") is False
    assert _is_repair_identity("interrupted-tool-result-c1-01", "c1") is False
    assert _is_repair_identity("interrupted-tool-result-c1-x", "c1") is False
    assert _is_repair_identity(f"interrupted-tool-result-c1-{2**53}", "c1") is False


def test_assert_content_kind_and_block() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="unclassified message content kind",
    ):
        _assert_content_kind("widget", "label")
    with pytest.raises(SessionFormatError, match="content must be an array"):
        _assert_content_kinds("x", "label")
    with pytest.raises(SessionFormatError, match="must be a JSON object"):
        _assert_content_block("x", "label")
    with pytest.raises(SessionFormatError, match="content must be an array"):
        _assert_content_block({"type": "tool-result"}, "label")
    nested = {"type": "tool-result", "content": [{"type": "text", "text": "x"}]}
    _assert_content_block(nested, "label")
    with pytest.raises(SessionFormatError, match="unclassified message content kind"):
        _assert_content_block(
            {"type": "tool-result", "content": [{"type": "widget"}]}, "label"
        )
    file_block = {
        "type": "file",
        "attachment": {"attachmentId": "a", "name": "f", "bytes": 3},
    }
    _assert_content_block(file_block, "label")
    with pytest.raises(SessionFormatError, match="has unexpected member"):
        _assert_content_block({**file_block, "extra": 1}, "label")
    with pytest.raises(SessionFormatError, match="attachment"):
        _assert_content_block({"type": "file", "attachment": "x"}, "label")
    with pytest.raises(SessionFormatError, match="requires attachmentId and name"):
        _assert_content_block(
            {
                "type": "file",
                "attachment": {"attachmentId": "", "name": "f", "bytes": 3},
            },
            "label",
        )
    with pytest.raises(SessionFormatError, match="requires attachmentId and name"):
        _assert_content_block(
            {
                "type": "file",
                "attachment": {"attachmentId": "a", "name": 3, "bytes": 3},
            },
            "label",
        )
    with pytest.raises(SessionFormatError, match="attachment bytes"):
        _assert_content_block(
            {
                "type": "file",
                "attachment": {"attachmentId": "a", "name": "f", "bytes": "x"},
            },
            "label",
        )


def test_assert_message_source_v2() -> None:
    _assert_message_source_v2({"source": {"kind": "user"}})
    with pytest.raises(SessionFormatError, match="message source"):
        _assert_message_source_v2({"source": "x"})
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="unclassified message source"
    ):
        _assert_message_source_v2({"source": {"kind": "vendor-x"}})
    _assert_message_source_v2(
        {"source": {"kind": "agent-message", "form": "relay", "senderSessionId": "s1"}}
    )
    with pytest.raises(SessionFormatError, match="has unexpected member"):
        _assert_message_source_v2(
            {
                "source": {
                    "kind": "agent-message",
                    "form": "relay",
                    "senderSessionId": "s1",
                    "extra": 1,
                }
            }
        )
    with pytest.raises(SessionFormatError, match="requires relay form"):
        _assert_message_source_v2(
            {
                "source": {
                    "kind": "agent-message",
                    "form": "direct",
                    "senderSessionId": "s1",
                }
            }
        )


def _v2_user_event(**overrides: object) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "user/message",
        "seq": 0,
        "time": 100,
        "surfaceOp": "append",
        "data": {
            "role": "user",
            "id": "m1",
            "content": [{"type": "text", "text": "hi"}],
            "source": {"kind": "user"},
        },
    }
    event.update(overrides)
    return event


def test_assert_event_v2_user_and_feedback() -> None:
    _assert_event_v2(_v2_user_event())
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="unclassified event"
    ):
        _assert_event_v2({"type": "vendor/widget", "seq": 0, "time": 100, "data": {}})
    with pytest.raises(SessionFormatError, match="ignorable must be true"):
        _assert_event_v2(_v2_user_event(ignorable=False))
    with pytest.raises(SessionFormatError, match="requires surfaceOp"):
        event = _v2_user_event()
        del event["surfaceOp"]
        _assert_event_v2(event)
    put = {
        "type": "feedback/message-put",
        "seq": 0,
        "time": 100,
        "data": {
            "sessionId": "s",
            "item": {
                "messageId": "m",
                "rating": "positive",
                "version": "1",
                "createdAt": 1,
                "updatedAt": 2,
            },
        },
    }
    _assert_event_v2(put)
    with pytest.raises(SessionFormatError, match="has unexpected member"):
        _assert_event_v2({**put, "data": {**put["data"], "extra": 1}})
    with pytest.raises(SessionFormatError, match="sessionId must be a string"):
        _assert_event_v2({**put, "data": {"sessionId": 3, "item": put["data"]["item"]}})
    with pytest.raises(SessionFormatError, match="invalid feedback rating"):
        bad = {
            **put,
            "data": {
                "sessionId": "s",
                "item": {**put["data"]["item"], "rating": "meh"},
            },
        }
        _assert_event_v2(bad)
    with pytest.raises(SessionFormatError, match="note must be a string"):
        bad = {
            **put,
            "data": {"sessionId": "s", "item": {**put["data"]["item"], "note": 3}},
        }
        _assert_event_v2(bad)
    with pytest.raises(SessionFormatError, match="messageId must be a string"):
        bad = {
            **put,
            "data": {"sessionId": "s", "item": {**put["data"]["item"], "messageId": 3}},
        }
        _assert_event_v2(bad)
    delete = {
        "type": "feedback/message-delete",
        "seq": 0,
        "time": 100,
        "data": {"sessionId": "s", "messageId": "m"},
    }
    _assert_event_v2(delete)
    with pytest.raises(SessionFormatError, match="messageId must be a string"):
        _assert_event_v2({**delete, "data": {"sessionId": "s", "messageId": 3}})


def test_assert_event_v2_owned_content_and_coordinates() -> None:
    event = _v2_user_event()
    event["data"]["content"] = [{"type": "widget"}]
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="unclassified message content kind",
    ):
        _assert_event_v2(event)
    event = _v2_user_event()
    event["data"]["source"] = {"kind": "vendor-x"}
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="unclassified message source"
    ):
        _assert_event_v2(event)
    assistant = {
        "type": "assistant/message",
        "seq": 0,
        "time": 100,
        "surfaceOp": "append",
        "data": {
            "turn": 0,
            "step": 1,
            "message": {
                "id": "m",
                "role": "assistant",
                "content": [],
                "source": {"kind": "user"},
            },
            "stream": [],
        },
    }
    with pytest.raises(SessionFormatError, match="turn must be positive"):
        _assert_event_v2(assistant)
    end_seed = {
        "type": "session/end-seed",
        "seq": 0,
        "time": 100,
        "data": {"inherited": False},
    }
    with pytest.raises(SessionFormatError, match="inherited must be true"):
        _assert_event_v2(end_seed)
    spliced = {
        "type": "agent/inbox/spliced",
        "seq": 0,
        "time": 100,
        "data": {
            "target": "t",
            "start": 0,
            "inserted": [
                {
                    "role": "user",
                    "id": "m",
                    "content": [],
                    "source": {"kind": "vendor-x"},
                }
            ],
        },
    }
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="unclassified message source"
    ):
        _assert_event_v2(spliced)
    spliced["data"]["inserted"] = "x"
    with pytest.raises(SessionFormatError, match="content must be an array"):
        _assert_owned_content(spliced, spliced["data"])
    summary = {
        "type": "compaction/summary",
        "seq": 0,
        "time": 100,
        "data": {
            "compactionId": "c",
            "summary": [{"type": "widget"}],
            "shadowedRange": {"start": 0, "end": 1},
            "shadowedSeqs": [0],
            "shadowedTokenCount": 1,
            "provider": "p",
            "model": "m",
        },
    }
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="unclassified message content kind",
    ):
        _assert_event_v2(summary)


def test_assert_event_v2_tool_not_started_repair() -> None:
    def tool_result(message_id: str) -> dict[str, Any]:
        return {
            "type": "tool/result",
            "seq": 0,
            "time": 100,
            "surfaceOp": "append",
            "data": {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": message_id,
                    "role": "tool",
                    "toolCallId": "c1",
                    "content": [{"type": "text", "text": "x"}],
                    "source": {"kind": "tool", "callId": "c1"},
                },
                "error": {"code": "TOOL_NOT_STARTED"},
            },
        }

    _assert_event_v2(tool_result("interrupted-tool-result-c1-0"))
    with pytest.raises(SessionFormatError, match="canonical historical message id"):
        _assert_event_v2(tool_result("interrupted-tool-result-c1-x"))


def test_assert_owned_content_streams() -> None:
    event = {
        "type": "assistant/message",
        "seq": 0,
        "time": 100,
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "m",
                "role": "assistant",
                "content": [],
                "source": {"kind": "user"},
            },
            "stream": [
                {
                    "type": "chunk",
                    "chunk": {
                        "type": "block-end",
                        "block": {"type": "text", "text": "x"},
                    },
                },
                {
                    "type": "chunk",
                    "chunk": {"type": "block-start", "blockType": "text"},
                },
                {"type": "note"},
            ],
        },
    }
    _assert_owned_content(event, event["data"])
    event["data"]["stream"] = "x"
    with pytest.raises(SessionFormatError, match="stream: content must be an array"):
        _assert_owned_content(event, event["data"])
    event["data"]["stream"] = [
        {"type": "chunk", "chunk": {"type": "block-start", "blockType": "widget"}}
    ]
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="unclassified message content kind",
    ):
        _assert_owned_content(event, event["data"])
    event["data"]["stream"] = [
        {"type": "chunk", "chunk": {"type": "block-end", "block": {"type": "widget"}}}
    ]
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="unclassified message content kind",
    ):
        _assert_owned_content(event, event["data"])
    event["data"]["stream"] = [{"type": "chunk"}]
    with pytest.raises(SessionFormatError, match="chunk"):
        _assert_owned_content(event, event["data"])


# ---------------------------------------------------------------------------
# Legacy normalizers (v0 -> v1)
# ---------------------------------------------------------------------------


def test_normalize_legacy_error_reason() -> None:
    failure = {"kind": "error", "step": 1, "failure": {"message": "boom", "code": "E1"}}
    assert _normalize_legacy_error_reason(failure, 0, "s") == {
        "kind": "error",
        "error": {"message": "boom", "code": "E1"},
    }
    with pytest.raises(SessionFormatError, match="malformed pre-react-loop"):
        _normalize_legacy_error_reason(
            {"kind": "error", "step": 1, "failure": {"message": 3, "code": "E1"}},
            0,
            "s",
        )
    flat = {"kind": "error", "step": 1, "message": "boom", "code": "E2"}
    assert _normalize_legacy_error_reason(flat, 0, "s") == {
        "kind": "error",
        "error": {"message": "boom", "code": "E2"},
    }
    assert _normalize_legacy_error_reason(
        {"kind": "error", "step": 1, "message": "boom"}, 0, "s"
    ) == {"kind": "error", "error": {"message": "boom", "code": "UNKNOWN"}}
    with pytest.raises(SessionFormatError, match="malformed pre-react-loop"):
        _normalize_legacy_error_reason(
            {"kind": "error", "step": 1, "message": 3}, 0, "s"
        )
    with pytest.raises(SessionFormatError, match="malformed pre-react-loop"):
        _normalize_legacy_error_reason(
            {"kind": "error", "step": 1, "message": "boom", "code": 3}, 0, "s"
        )


def test_normalize_legacy_turn_end() -> None:
    def event(reason: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "turn/end",
            "seq": 0,
            "time": 1,
            "data": {"turn": 1, "reason": reason},
        }

    assert _normalize_legacy_turn_end(event({"kind": "completed"}), "s") == event(
        {"kind": "completed"}
    )
    with pytest.raises(SessionFormatError, match="malformed pre-react-loop"):
        _normalize_legacy_turn_end(
            {
                "type": "turn/end",
                "seq": 0,
                "time": 1,
                "data": {"turn": 0, "reason": {"kind": "completed"}},
            },
            "s",
        )
    with pytest.raises(SessionFormatError, match="malformed pre-react-loop"):
        _normalize_legacy_turn_end(event({"kind": 3}), "s")
    assert _normalize_legacy_turn_end(
        event({"kind": "aborted", "reason": {"kind": "disposed"}}), "s"
    ) == event({"kind": "aborted", "reason": {"kind": "disposed"}})
    assert _normalize_legacy_turn_end(event({"kind": "disposed"}), "s")["data"][
        "reason"
    ] == {
        "kind": "aborted",
        "reason": {"kind": "disposed"},
    }
    assert _normalize_legacy_turn_end(
        event({"kind": "error", "error": {"message": "x", "code": "E"}}), "s"
    ) == event({"kind": "error", "error": {"message": "x", "code": "E"}})
    assert _normalize_legacy_turn_end(
        event({"kind": "error", "step": 1, "message": "boom"}), "s"
    )["data"]["reason"] == {
        "kind": "error",
        "error": {"message": "boom", "code": "UNKNOWN"},
    }
    assert _normalize_legacy_turn_end(event({"kind": "weird"}), "s") == event(
        {"kind": "weird"}
    )
    assert (
        _normalize_legacy_turn_end(
            {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}, "s"
        )["type"]
        == "turn/start"
    )


def test_normalize_legacy_steering_and_request_header() -> None:
    wrapped = {
        "type": "steering/message",
        "seq": 0,
        "time": 1,
        "data": {
            "turn": 1,
            "message": {
                "role": "user",
                "id": "m",
                "content": [],
                "source": {"kind": "user"},
            },
        },
    }
    normalized = _normalize_legacy_steering(wrapped, "s")
    assert normalized["type"] == "user/message"
    assert normalized["data"] == wrapped["data"]["message"]
    assert (
        _normalize_legacy_steering(
            {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}, "s"
        )["type"]
        == "turn/start"
    )
    header = {
        "type": "request/header",
        "seq": 0,
        "time": 1,
        "data": {"header": {"messagePrefix": ["p"], "reason": {"kind": "start"}}},
    }
    assert _normalize_legacy_request_header(header, "s")["data"]["header"] == {
        "reason": {"kind": "start"}
    }
    assert _normalize_legacy_request_header(
        {
            "type": "request/header",
            "seq": 0,
            "time": 1,
            "data": {"header": {"reason": {"kind": "start"}}},
        },
        "s",
    )["data"]["header"] == {"reason": {"kind": "start"}}
    with pytest.raises(
        SessionFormatError, match="malformed request/header messagePrefix"
    ):
        _normalize_legacy_request_header(
            {
                "type": "request/header",
                "seq": 0,
                "time": 1,
                "data": {"header": {"messagePrefix": "p"}},
            },
            "s",
        )
    assert (
        _normalize_legacy_request_header(
            {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}, "s"
        )["type"]
        == "turn/start"
    )


def test_normalize_legacy_retry() -> None:
    retry_ids: dict[str, str] = {}
    event = {
        "type": "llm/retry",
        "seq": 0,
        "time": 1,
        "data": {
            "turn": 1,
            "step": 1,
            "provider": "p",
            "mode": "m",
            "policyKey": "k",
            "retry": 1,
            "delayMs": 5,
            "failure": {},
        },
    }
    assert (
        _normalize_legacy_retry(event, "s", retry_ids)["data"]["retryId"]
        == "legacy-retry:s:0"
    )
    assert retry_ids
    second = {**event, "seq": 1}
    assert (
        _normalize_legacy_retry(second, "s", retry_ids)["data"]["retryId"]
        == "legacy-retry:s:0"
    )
    explicit = {**event, "seq": 2, "data": {**event["data"], "retryId": "r-9"}}
    assert _normalize_legacy_retry(explicit, "s", retry_ids) is explicit
    null_id = {**event, "seq": 3, "data": {**event["data"], "retryId": None}}
    assert _normalize_legacy_retry(null_id, "s", retry_ids) is null_id
    assert (
        _normalize_legacy_retry(
            {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
            "s",
            retry_ids,
        )["type"]
        == "turn/start"
    )


def test_normalize_legacy_compaction_state() -> None:
    state = _LegacyCompactionState()
    start = {"type": "compaction/start", "seq": 0, "time": 1, "data": {"turn": 1}}
    assert (
        _normalize_legacy_compaction(start, "s", state)["data"]["compactionId"]
        == "legacy-compaction:s:0"
    )
    assert state.compaction_id == "legacy-compaction:s:0"
    summary = {
        "type": "compaction/summary",
        "seq": 1,
        "time": 2,
        "data": {"summary": []},
    }
    assert (
        _normalize_legacy_compaction(summary, "s", state)["data"]["compactionId"]
        == "legacy-compaction:s:0"
    )
    end = {"type": "compaction/end", "seq": 2, "time": 3, "data": {}}
    assert (
        _normalize_legacy_compaction(end, "s", state)["data"]["compactionId"]
        == "legacy-compaction:s:0"
    )
    assert state.compaction_id is None
    later = {"type": "compaction/summary", "seq": 3, "time": 4, "data": {"summary": []}}
    assert _normalize_legacy_compaction(later, "s", state) is later
    existing = {
        "type": "compaction/start",
        "seq": 4,
        "time": 5,
        "data": {"turn": 1, "compactionId": "c9"},
    }
    assert _normalize_legacy_compaction(existing, "s", state) is existing
    assert state.compaction_id == "c9"
    seed = {
        "type": "session/end-seed",
        "seq": 5,
        "time": 6,
        "data": {"inherited": True},
    }
    assert _normalize_legacy_compaction(seed, "s", state) is seed
    assert state.compaction_id is None
    compact_user = {
        "type": "user/message",
        "seq": 6,
        "time": 7,
        "data": {
            "role": "user",
            "id": "m",
            "content": [],
            "source": {"kind": "plugin", "plugin": "compact"},
        },
    }
    state.compaction_id = "c7"
    normalized = _normalize_legacy_compaction(compact_user, "s", state)
    assert normalized["data"]["source"]["compactionId"] == "c7"
    plain_user = {
        **compact_user,
        "data": {**compact_user["data"], "source": {"kind": "user"}},
    }
    assert _normalize_legacy_compaction(plain_user, "s", state) is plain_user
    assert (
        _normalize_legacy_compaction(
            {"type": "turn/start", "seq": 7, "time": 8, "data": {"turn": 1}}, "s", state
        )["type"]
        == "turn/start"
    )


def test_replacement_start() -> None:
    assert (
        _replacement_start({"type": "tool/result", "seq": 0, "time": 1, "data": {}})
        is None
    )
    assert (
        _replacement_start(
            {
                "type": "tool/result",
                "seq": 0,
                "time": 1,
                "surfaceOp": "append",
                "data": {},
            }
        )
        is None
    )
    assert (
        _replacement_start(
            {
                "type": "tool/result",
                "seq": 5,
                "time": 1,
                "surfaceOp": {"op": "replace", "start": 2, "end": 3},
                "data": {},
            }
        )
        == 2
    )


def test_normalize_legacy_message() -> None:
    flat_user = {
        "type": "user/message",
        "seq": 0,
        "time": 1,
        "data": {"content": [], "source": {"kind": "user"}},
    }
    normalized = _normalize_legacy_message(flat_user, "s", {})
    assert normalized["data"]["id"] == "legacy-message:s:0"
    assert normalized["data"]["role"] == "user"
    with_id = {**flat_user, "data": {**flat_user["data"], "id": "m"}}
    assert _normalize_legacy_message(with_id, "s", {}) is with_id
    flat_assistant = {
        "type": "assistant/message",
        "seq": 1,
        "time": 2,
        "data": {"turn": 1, "step": 1, "content": [], "provenance": {"provider": "p"}},
    }
    normalized = _normalize_legacy_message(flat_assistant, "s", {})
    assert normalized["data"]["message"]["role"] == "assistant"
    assert normalized["data"]["message"]["source"] == {"provider": "p", "kind": "model"}
    assert normalized["data"]["message"]["id"] == "legacy-message:s:1"
    wrapped_assistant = {
        **flat_assistant,
        "data": {**flat_assistant["data"], "message": {}},
    }
    assert _normalize_legacy_message(wrapped_assistant, "s", {}) is wrapped_assistant
    flat_tool = {
        "type": "tool/result",
        "seq": 2,
        "time": 3,
        "data": {
            "turn": 1,
            "step": 1,
            "callId": "c1",
            "content": [{"type": "text", "text": "3"}],
            "isError": False,
        },
    }
    normalized = _normalize_legacy_message(flat_tool, "s", {})
    message = normalized["data"]["message"]
    assert message["id"] == "legacy-message:s:2"
    assert message["role"] == "user"
    assert message["content"] == [
        {
            "type": "tool-result",
            "toolCallId": "c1",
            "content": [{"type": "text", "text": "3"}],
            "isError": False,
        }
    ]
    assert message["source"] == {"kind": "tool", "callId": "c1"}
    replaced = {
        **flat_tool,
        "seq": 5,
        "surfaceOp": {"op": "replace", "start": 2, "end": 3},
    }
    normalized = _normalize_legacy_message(replaced, "s", {2: "m-orig"})
    assert normalized["data"]["message"]["id"] == "m-orig"
    with pytest.raises(
        SessionFormatError, match="replacement cites a message without identity"
    ):
        _normalize_legacy_message(replaced, "s", {})
    bad_call = {**flat_tool, "data": {**flat_tool["data"], "callId": 3}}
    assert _normalize_legacy_message(bad_call, "s", {}) is bad_call
    no_content = {
        **flat_tool,
        "data": {k: v for k, v in flat_tool["data"].items() if k != "content"},
    }
    assert _normalize_legacy_message(no_content, "s", {}) is no_content
    assert (
        _normalize_legacy_message(
            {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}, "s", {}
        )["type"]
        == "turn/start"
    )


# ---------------------------------------------------------------------------
# Stream records, packed runs, v1 -> v2 message handling
# ---------------------------------------------------------------------------


def test_append_stream_record_merges_and_splits() -> None:
    group = _AttemptGroup(1, 1)
    _append_stream_record(
        group,
        {
            "type": "text-chunks",
            "time0": 100,
            "index": 0,
            "dt": [1],
            "texts": ["a", "b"],
        },
        101,
    )
    _append_stream_record(
        group,
        {
            "type": "text-chunks",
            "time0": 102,
            "index": 0,
            "dt": [1],
            "texts": ["c", "d"],
        },
        103,
    )
    assert group.stream[0]["record"]["texts"] == ["a", "b", "c", "d"]
    assert group.stream[0]["record"]["dt"] == [1, 1, 1]
    _append_stream_record(
        group,
        {"type": "text-chunks", "time0": 104, "index": 1, "dt": [], "texts": ["e"]},
        104,
    )
    assert len(group.stream) == 2
    _append_stream_record(
        group, {"type": "chunk", "time": 105, "chunk": {"type": "block-start"}}, 105
    )
    assert len(group.stream) == 3
    _append_stream_record(
        group,
        {
            "type": "tool-call-chunks",
            "time0": 106,
            "index": 0,
            "dt": [],
            "id": "c1",
            "name": "f",
            "args": ["{}"],
        },
        106,
    )
    _append_stream_record(
        group,
        {
            "type": "tool-call-chunks",
            "time0": 107,
            "index": 0,
            "dt": [1],
            "id": "c1",
            "name": "f",
            "args": ["{}"],
        },
        108,
    )
    assert group.stream[-1]["record"]["args"] == ["{}", "{}"]
    _append_stream_record(
        group,
        {
            "type": "tool-call-chunks",
            "time0": 109,
            "index": 0,
            "dt": [],
            "id": "c2",
            "name": "f",
            "args": ["{}"],
        },
        109,
    )
    assert len(group.stream) == 5
    _append_stream_record(
        group,
        {
            "type": "tool-call-chunks",
            "time0": 110,
            "index": 0,
            "dt": [],
            "id": "c2",
            "name": "g",
            "args": ["{}"],
        },
        110,
    )
    assert len(group.stream) == 6
    _append_stream_record(
        group,
        {
            "type": "text-chunks",
            "time0": 2**53 - 1,
            "index": 0,
            "dt": [],
            "texts": ["z"],
        },
        2**53 - 1,
    )
    assert len(group.stream) == 7


def _v1_stage() -> tuple[object, _Collector]:
    migration = SESSION_FORMAT_V1_TO_V2
    logical = {
        "version": 1,
        "id": "v1",
        "createdAt": 1,
        "isSeeded": False,
        "delegationDepth": 0,
    }
    stage = migration.create_stage(
        SessionFormatMigrationStageInput(
            source_header=dict(logical),
            target_header=migration.migrate_header(dict(logical)),
            source_inherited_event_count=0,
            source_kind="decoded",
        )
    )
    return stage, _Collector()


def _chunk_row(
    seq: int, time: int, chunk: dict[str, Any], **extra: object
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": "assistant/chunk",
        "seq": seq,
        "time": time,
        "data": {"turn": 1, "step": 1, "chunk": chunk},
    }
    row.update(extra)
    return row


def _packed_run(
    first_seq: int, turn: int, step: int, texts: list[str]
) -> PackedAssistantChunkRun:
    return PackedAssistantChunkRun(
        run_type="released-assistant-chunks",
        first_seq=first_seq,
        event_count=len(texts),
        stream={
            "type": "text-chunks",
            "time0": 100,
            "index": 0,
            "dt": [1] * (len(texts) - 1),
            "texts": texts,
        },
        turn=turn,
        step=step,
        last_seq=first_seq + len(texts) - 1,
        last_time=100 + len(texts) - 1,
    )


def test_transform_released_run_paths() -> None:
    stage, out = _v1_stage()
    stage.transform_run(_packed_run(0, 1, 1, ["a", "b"]), out)
    assert stage.finish(out) == 0
    assert [event["type"] for event in out.values] == ["assistant/attempt"]
    assert out.values[0]["data"]["stream"][0]["texts"] == ["a", "b"]
    stage, out = _v1_stage()
    stage.transform_event(
        _chunk_row(0, 100, {"type": "text-delta", "index": 0, "text": "a"}), out
    )
    stage.transform_event(
        _chunk_row(1, 101, {"type": "finish", "reason": {"kind": "stop"}}), out
    )
    stage.transform_run(_packed_run(2, 2, 1, ["c"]), out)
    assert stage.finish(out) == 0
    assert [event["type"] for event in out.values] == [
        "assistant/attempt",
        "assistant/attempt",
    ]
    assert out.values[0]["data"]["stream"][0]["texts"] == ["a"]
    assert out.values[1]["data"]["stream"][0]["texts"] == ["c"]
    stage, out = _v1_stage()
    stage.transform_event(
        _chunk_row(0, 100, {"type": "text-delta", "index": 0, "text": "a"}), out
    )
    stage.transform_run(_packed_run(1, 1, 1, ["b", "c"]), out)
    assert stage.finish(out) == 0
    assert len(out.values) == 1
    assert out.values[0]["data"]["stream"][0]["texts"] == ["a", "b", "c"]
    stage, out = _v1_stage()
    plain = SessionFormatEventRun(
        run_type="plain",
        first_seq=0,
        event_count=1,
        expand_events=[
            {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}
        ],
    )
    stage.transform_run(plain, out)
    assert [event["type"] for event in out.values] == ["turn/start"]


def test_transform_message_branches() -> None:
    header = _header(1, "v1")
    message = {
        "type": "assistant/message",
        "seq": 1,
        "time": 101,
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "m",
                "role": "assistant",
                "content": [],
                "source": {"kind": "user"},
            },
        },
    }
    _, _, values = _run_stage(
        SESSION_FORMAT_V1_TO_V2,
        header,
        0,
        [
            _chunk_row(0, 100, {"type": "text-delta", "index": 0, "text": "a"}),
            {**message, "data": {**message["data"], "step": 2}},
        ],
    )
    assert [event["type"] for event in values] == [
        "assistant/attempt",
        "assistant/message",
    ]
    with pytest.raises(SessionFormatUnsupportedMigrationError, match="does not cite"):
        _run_stage(
            SESSION_FORMAT_V1_TO_V2,
            header,
            0,
            [
                _chunk_row(0, 100, {"type": "text-delta", "index": 0, "text": "a"}),
                message,
            ],
        )
    _, _, values = _run_stage(
        SESSION_FORMAT_V1_TO_V2,
        header,
        0,
        [
            _chunk_row(0, 100, {"type": "text-delta", "index": 0, "text": "a"}),
            {**message, "sourceEventSeqs": []},
        ],
    )
    assert [event["type"] for event in values] == [
        "assistant/attempt",
        "assistant/message",
    ]
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="not one complete ordered attempt"
    ):
        _run_stage(
            SESSION_FORMAT_V1_TO_V2,
            header,
            0,
            [message | {"sourceEventSeqs": [0]}],
        )


# ---------------------------------------------------------------------------
# v3 content migration, references, PTC rename
# ---------------------------------------------------------------------------


def test_migrate_v3_chunk_variants() -> None:
    assert _migrate_v3_chunk("x", "s") == "x"
    assert _migrate_v3_chunk({"type": "text-delta", "index": 0}, "s") == {
        "type": "text-delta",
        "index": 0,
    }
    assert _migrate_v3_chunk({"type": "block-start", "blockType": "text"}, "s") == {
        "type": "block-start",
        "blockType": "text",
    }
    assert _migrate_v3_chunk({"type": "block-start", "blockType": "widget"}, "s") == {
        "type": "block-start",
        "blockType": "plugin:widget",
    }
    assert _migrate_v3_chunk(
        {"type": "block-end", "block": {"type": "widget", "text": "x"}}, "s"
    ) == {
        "type": "block-end",
        "block": {"type": "plugin:widget", "text": "x"},
    }
    with pytest.raises(SessionFormatError, match="blockType must be a string"):
        _migrate_v3_chunk({"type": "block-start", "blockType": 3}, "s")


def test_migrate_v3_event_content() -> None:
    summary = {
        "type": "compaction/summary",
        "seq": 0,
        "time": 100,
        "data": {
            "compactionId": "c",
            "summary": [{"type": "widget", "text": "x"}],
            "rawOutput": [{"type": "widget", "text": "y"}],
            "shadowedRange": {"start": 0, "end": 0},
            "shadowedSeqs": [0],
            "shadowedTokenCount": 1,
            "provider": "p",
            "model": "m",
        },
    }
    migrated = migrate_v3_event_content(summary)
    assert migrated["data"]["summary"][0]["type"] == "plugin:widget"
    assert migrated["data"]["rawOutput"][0]["type"] == "plugin:widget"
    dispatch = {
        "type": "tool/ptc-dispatch",
        "seq": 0,
        "time": 100,
        "data": {"content": [{"type": "widget"}]},
    }
    assert (
        migrate_v3_event_content(dispatch)["data"]["content"][0]["type"]
        == "plugin:widget"
    )
    queued = {
        "type": "team/message/queued",
        "seq": 0,
        "time": 100,
        "data": {
            "message": {
                "id": "m",
                "role": "user",
                "content": [{"type": "widget"}],
                "source": {"kind": "user"},
            }
        },
    }
    assert (
        migrate_v3_event_content(queued)["data"]["message"]["content"][0]["type"]
        == "plugin:widget"
    )
    with pytest.raises(SessionFormatUnsupportedMigrationError, match="deferLoading"):
        migrate_v3_event_content(
            {
                "type": "request/header",
                "seq": 0,
                "time": 100,
                "data": {"header": {"tools": [{"name": "x", "deferLoading": True}]}},
            }
        )
    message = {
        "type": "assistant/message",
        "seq": 0,
        "time": 100,
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "m",
                "role": "assistant",
                "content": [],
                "source": {"kind": "user"},
            },
            "stream": [
                "note",
                {"type": "note"},
                {
                    "type": "chunk",
                    "chunk": {"type": "block-start", "blockType": "widget"},
                },
                {
                    "type": "chunk",
                    "chunk": {"type": "block-end", "block": {"type": "widget"}},
                },
                {"type": "chunk", "chunk": "x"},
            ],
        },
    }
    migrated = migrate_v3_event_content(message)
    assert migrated["data"]["stream"][2]["chunk"]["blockType"] == "plugin:widget"
    assert migrated["data"]["stream"][3]["chunk"]["block"]["type"] == "plugin:widget"
    assert migrated["data"]["stream"][0] == "note"
    assert (
        migrate_v3_event_content(
            {"type": "turn/start", "seq": 0, "time": 100, "data": "x"}
        )["data"]
        == "x"
    )


def test_rename_ptc_event() -> None:
    assert (
        _rename_ptc_event(
            {"type": "tool/code-dispatch-start", "seq": 0, "time": 1, "data": {}}
        )["type"]
        == "tool/ptc-dispatch-start"
    )
    assert (
        _rename_ptc_event(
            {"type": "tool/code-dispatch", "seq": 0, "time": 1, "data": {}}
        )["type"]
        == "tool/ptc-dispatch"
    )
    preset = {
        "type": "agent-preset/selected",
        "seq": 0,
        "time": 1,
        "data": {"agentPreset": "ptc"},
    }
    assert _rename_ptc_event(preset) is preset
    user = {
        "type": "user/message",
        "seq": 0,
        "time": 1,
        "data": {"role": "user", "id": "m", "content": [], "source": {"kind": "user"}},
    }
    assert _rename_ptc_event(user) is user
    spliced = {
        "type": "agent/inbox/spliced",
        "seq": 0,
        "time": 1,
        "data": {
            "target": "t",
            "start": 0,
            "inserted": [
                {
                    "role": "user",
                    "id": "m",
                    "content": [],
                    "source": {"kind": "plugin", "plugin": "tools-code-mode"},
                }
            ],
        },
    }
    assert (
        _rename_ptc_event(spliced)["data"]["inserted"][0]["source"]["plugin"]
        == "tools-ptc"
    )
    spliced["data"]["inserted"] = "x"
    assert _rename_ptc_event(spliced) is spliced
    spliced["data"]["inserted"] = ["x"]
    assert _rename_ptc_event(spliced) is spliced
    assert (
        _rename_ptc_event(
            {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}
        )["type"]
        == "turn/start"
    )


def test_remap_v3_reference_branches() -> None:
    header = _header(2, "v2")
    opening = [
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}},
        {"type": "step/start", "seq": 1, "time": 101, "data": {"turn": 1, "step": 1}},
    ]
    _, _, values = _run_stage(
        SESSION_FORMAT_V2_TO_V3,
        header,
        0,
        opening
        + [
            {
                "type": "command/done",
                "seq": 2,
                "time": 102,
                "data": {"commandId": "c", "kind": "k", "sourceEventSeq": 0},
            },
        ],
    )
    assert values[3]["data"]["sourceEventSeq"] == 0
    with pytest.raises(SessionFormatError, match="must name an earlier source event"):
        _run_stage(
            SESSION_FORMAT_V2_TO_V3,
            header,
            0,
            opening
            + [
                {
                    "type": "command/done",
                    "seq": 2,
                    "time": 102,
                    "data": {"commandId": "c", "kind": "k", "sourceEventSeq": 5},
                }
            ],
        )
    with pytest.raises(SessionFormatError, match="must be an array"):
        _run_stage(
            SESSION_FORMAT_V2_TO_V3,
            header,
            0,
            opening
            + [
                {
                    "type": "session/title",
                    "seq": 2,
                    "time": 102,
                    "data": {"title": "t", "messageSeqs": "x", "source": "llm"},
                }
            ],
        )
    # The v3 -> v4 reference remap (including image/offload targets, which
    # the v2 stage never sees) passes through the stage unchanged: the v3
    # stage never shifts seqs, so the helper's early return applies there.
    # The remap body itself is exercised directly with a shifted seq.
    v3_header = _header(3, "v3")
    v3_opening = [{"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}]
    _, _, values = _run_stage(
        _v3_migration([]),
        v3_header,
        0,
        v3_opening
        + [
            {
                "type": "image/offload",
                "seq": 1,
                "time": 101,
                "data": {"targets": [{"seq": 0, "attachmentId": "a"}]},
            },
            {
                "type": "command/done",
                "seq": 2,
                "time": 102,
                "data": {"commandId": "c", "kind": "k", "sourceEventSeq": 0},
            },
            {
                "type": "compaction/summary",
                "seq": 3,
                "time": 103,
                "data": {
                    "compactionId": "c",
                    "summary": [],
                    "shadowedRange": {"start": 0, "end": 1},
                    "shadowedSeqs": [0, 1],
                    "shadowedTokenCount": 1,
                    "provider": "p",
                    "model": "m",
                },
            },
            {
                "type": "session/title",
                "seq": 4,
                "time": 104,
                "data": {"title": "t", "messageSeqs": [0], "source": "llm"},
            },
        ],
    )
    assert values[1]["data"]["targets"][0]["seq"] == 0
    assert values[2]["data"]["sourceEventSeq"] == 0
    assert values[3]["data"]["shadowedRange"] == {"start": 0, "end": 1}
    assert values[3]["data"]["shadowedSeqs"] == [0, 1]
    assert values[4]["data"]["messageSeqs"] == [0]
    shifted = [0, 3, 5]
    offload = {
        "type": "image/offload",
        "seq": 2,
        "time": 103,
        "data": {"targets": [{"seq": 1, "attachmentId": "a"}]},
    }
    assert _remap_v3_references(offload, 5, shifted)["data"]["targets"][0]["seq"] == 3
    done = {
        "type": "command/done",
        "seq": 2,
        "time": 103,
        "data": {"commandId": "c", "kind": "k", "sourceEventSeq": 1},
    }
    assert _remap_v3_references(done, 5, shifted)["data"]["sourceEventSeq"] == 3
    summary = {
        "type": "compaction/summary",
        "seq": 2,
        "time": 103,
        "data": {"shadowedRange": {"start": 0, "end": 1}, "shadowedSeqs": [0, 1]},
    }
    remapped = _remap_v3_references(summary, 5, shifted)
    assert remapped["data"]["shadowedRange"] == {"start": 0, "end": 3}
    assert remapped["data"]["shadowedSeqs"] == [0, 3]
    title = {
        "type": "session/title",
        "seq": 2,
        "time": 103,
        "data": {"title": "t", "messageSeqs": [1], "source": "llm"},
    }
    assert _remap_v3_references(title, 5, shifted)["data"]["messageSeqs"] == [3]
    replaced = {
        "type": "user/message",
        "seq": 2,
        "time": 103,
        "surfaceOp": {"op": "replace", "startSeq": 0, "endSeq": 1},
        "sourceEventSeqs": [1],
        "data": {"role": "user", "id": "m", "content": [], "source": {"kind": "user"}},
    }
    remapped = _remap_v3_references(replaced, 5, shifted)
    assert remapped["surfaceOp"] == {"op": "replace", "startSeq": 0, "endSeq": 3}
    assert remapped["sourceEventSeqs"] == [3]
    assert _remap_v3_references(done, 2, shifted) is done
    with pytest.raises(SessionFormatError, match="targets must be an array"):
        _remap_v3_references(
            {"type": "image/offload", "seq": 2, "time": 103, "data": {"targets": "x"}},
            5,
            shifted,
        )
    with pytest.raises(SessionFormatError, match="must name an earlier source event"):
        _remap_v3_references(
            {
                "type": "image/offload",
                "seq": 2,
                "time": 103,
                "data": {"targets": [{"seq": 5}]},
            },
            5,
            shifted,
        )
    with pytest.raises(SessionFormatError, match="must be an array"):
        _remap_v3_references(
            {
                "type": "session/title",
                "seq": 2,
                "time": 103,
                "data": {"title": "t", "messageSeqs": "x", "source": "llm"},
            },
            5,
            shifted,
        )
    # The v2 -> v3 remap helper passes image/offload data through untouched
    # (it has no v2 reference fields); only its seq is rewritten.
    offload = {
        "type": "image/offload",
        "seq": 1,
        "time": 103,
        "data": {"targets": [{"seq": 0, "attachmentId": "a"}]},
    }
    assert _remap_event_v2_to_v3(offload, 5, [0, 1])["seq"] == 5
    assert _remap_event_v2_to_v3(offload, 5, [0, 1])["data"]["targets"][0]["seq"] == 0


# ---------------------------------------------------------------------------
# Decoder gap-on-turn/end branches
# ---------------------------------------------------------------------------


def test_decoder_gap_on_turn_end() -> None:
    decoder = _v0_decoder("recoverable")
    out = _Collector()
    decoder.decode_row(
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}, out
    )
    with pytest.raises(SessionFormatError, match="seq gap"):
        decoder.decode_row(
            {
                "type": "turn/end",
                "seq": 5,
                "time": 101,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
            out,
        )
    decoder = _v2_decoder("recoverable")
    out = _Collector()
    decoder.decode_row(
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}, out
    )
    with pytest.raises(SessionFormatError, match="seq gap"):
        decoder.decode_row(
            {
                "type": "turn/end",
                "seq": 5,
                "time": 101,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
            out,
        )


# ---------------------------------------------------------------------------
# Chunk 11: header / payload / surface internals
# ---------------------------------------------------------------------------


def _logical_header(version: int, session_id: str = "s", **extra: object) -> dict:
    """One logical header (no physical 'type' member) for direct validation."""
    header: dict = {
        "version": version,
        "id": session_id,
        "createdAt": 1000,
        "isSeeded": False,
        "delegationDepth": 0,
    }
    header.update(extra)
    return header


def test_assert_v0_v1_header_rejections() -> None:
    base = _logical_header(0)
    with pytest.raises(SessionFormatError, match="expected format v1 header"):
        assert_v0_v1_header(base, 1)
    with pytest.raises(SessionFormatError, match="id must be a string"):
        assert_v0_v1_header({**base, "id": 7}, 0)
    with pytest.raises(SessionFormatError, match="isSeeded must be boolean"):
        assert_v0_v1_header({**base, "isSeeded": 1}, 0)
    with pytest.raises(SessionFormatError, match="cwd must be absolute"):
        assert_v0_v1_header({**base, "cwd": "relative/path"}, 0)
    with pytest.raises(SessionFormatError, match="parentSession must be a string"):
        assert_v0_v1_header({**base, "parentSession": 3}, 0)
    with pytest.raises(SessionFormatError, match='origin must be "subagent"'):
        assert_v0_v1_header({**base, "origin": "user"}, 0)
    assert assert_v0_v1_header(base, 0)["id"] == "s"


def test_assert_v3_header_wrong_version() -> None:
    with pytest.raises(SessionFormatError, match="expected format v3 header"):
        assert_v3_header(_logical_header(4))


def test_assert_released_event_payload_branches() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="contains unknown historical event type",
    ):
        assert_released_event_payload({"type": "vendor/x", "seq": 0, "data": {}}, 0)
    descriptor = {
        "type": "subagent/descriptor",
        "seq": 0,
        "data": {"version": 2},
    }
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="uses unsupported descriptor version 2",
    ):
        assert_released_event_payload(descriptor, 0)
    # Version 1 tolerates a pre-v3 descriptor: the early return, no exact keys.
    assert_released_event_payload(descriptor, 1)
    # Version 1 accepts the optional sessionFormatVersion on a delivery marker.
    assert_released_event_payload(
        {
            "type": "session-log-deepseek/delivery-accepted",
            "seq": 0,
            "data": {"sessionId": "s", "throughSeq": 0, "sessionFormatVersion": 1},
        },
        1,
    )
    with pytest.raises(
        SessionFormatError, match="compactionId must be a non-empty string"
    ):
        assert_released_event_payload(
            {
                "type": "compaction/start",
                "seq": 0,
                "data": {"compactionId": "", "turn": 1},
            },
            0,
        )


def test_assert_released_surface_metadata_branches() -> None:
    with pytest.raises(SessionFormatError, match="retains obsolete chunk references"):
        assert_released_surface_metadata(
            {"sourceEventSeqs": [0]}, 1, "assistant/message", "forbid-assistant"
        )
    with pytest.raises(SessionFormatError, match="sourceEventSeqs must be an array"):
        assert_released_surface_metadata(
            {"sourceEventSeqs": 0}, 1, "user/message", "allow"
        )
    with pytest.raises(
        SessionFormatError, match="sourceEventSeqs must be unique earlier seqs"
    ):
        assert_released_surface_metadata(
            {"sourceEventSeqs": [0, 0]}, 2, "user/message", "allow"
        )
    with pytest.raises(
        SessionFormatError, match="sourceEventSeqs must be unique earlier seqs"
    ):
        assert_released_surface_metadata(
            {"sourceEventSeqs": [2]}, 2, "user/message", "allow"
        )
    with pytest.raises(SessionFormatError, match="sourceEventSeqs must be non-empty"):
        assert_released_surface_metadata(
            {"sourceEventSeqs": []}, 1, "user/message", "allow"
        )
    # An empty source list is allowed for assistant/message.
    assert_released_surface_metadata(
        {"sourceEventSeqs": []}, 1, "assistant/message", "allow"
    )
    with pytest.raises(SessionFormatError, match="surfaceOp must replace"):
        assert_released_surface_metadata(
            {"surfaceOp": {"op": "move", "start": 0, "end": 0}},
            2,
            "user/message",
            "allow",
        )
    with pytest.raises(SessionFormatError, match="has an invalid surface replacement"):
        assert_released_surface_metadata(
            {"surfaceOp": {"op": "replace", "start": 0, "end": 5}},
            5,
            "user/message",
            "allow",
        )
    assert_released_surface_metadata(
        {"surfaceOp": {"op": "replace", "start": 0, "end": 1}},
        5,
        "user/message",
        "allow",
    )


def test_accepted_through_skips_non_delivery() -> None:
    assert (
        accepted_through(
            [
                {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
                {
                    "type": "session-log-deepseek/delivery-accepted",
                    "seq": 1,
                    "time": 2,
                    "data": {"sessionId": "s", "throughSeq": 0},
                },
            ],
            0,
            "s",
        )
        == 0
    )


def test_parse_session_format_log_filename_non_digit() -> None:
    assert parse_session_format_log_filename("session.vx.jsonl") is None


def test_decode_physical_header_rejections() -> None:
    header = _header(0, "s")
    with pytest.raises(
        SessionFormatError, match="expected released v0 physical Session header"
    ):
        _V01Codec(0).create_decoder({**header, "type": "event"}, "strict")
    with pytest.raises(SessionFormatError, match="header id must be a string"):
        _V01Codec(0).create_decoder({**header, "id": 7}, "strict")
    with pytest.raises(SessionFormatError, match="parentSession must be a string"):
        _V01Codec(0).create_decoder({**header, "parentSession": 3}, "strict")
    with pytest.raises(SessionFormatError, match='header origin must be "subagent"'):
        _V01Codec(0).create_decoder({**header, "origin": "user"}, "strict")


# ---------------------------------------------------------------------------
# Chunk 11: v4 developer / retired / codec encode branches
# ---------------------------------------------------------------------------


def test_v4_developer_data_spliced_raw_output_and_stream() -> None:
    # Spliced messages without a developer role still get their content
    # blocks checked.
    assert_v4_developer_data(
        {
            "type": "agent/inbox/spliced",
            "seq": 0,
            "time": 100,
            "data": {
                "inserted": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]}
                ]
            },
        }
    )
    # Compaction rawOutput is checked alongside the summary.
    assert_v4_developer_data(
        {
            "type": "compaction/summary",
            "seq": 0,
            "time": 100,
            "data": {"rawOutput": [{"type": "text", "text": "x"}]},
        }
    )
    # Stream entries that are not chunk objects are skipped, not rejected.
    assert_v4_developer_data(
        {
            "type": "assistant/message",
            "seq": 0,
            "time": 100,
            "data": {"stream": ["x", {"type": "text"}]},
        }
    )


def test_assert_v4_retired_syntax_non_object_data() -> None:
    assert_v4_retired_syntax({"type": "turn/start", "seq": 0, "time": 1, "data": "x"})


def test_v3_and_v4_codec_decode_header_wrong_version() -> None:
    with pytest.raises(
        SessionFormatError, match="expected format v3 physical Session header"
    ):
        _V3Codec().decode_header(_header(4, "s"))
    with pytest.raises(SessionFormatError, match="expected format v4 physical header"):
        _V4Codec().decode_header(_header(3, "s"))


def test_v3_codec_encode_event() -> None:
    row = _V3Codec().encode_event(
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}
    )
    assert row["type"] == "turn/start"
    assert row["seq"] == 0
    assert row["data"] == {"turn": 1}


def test_v4_codec_encode_event_developer_ignorable() -> None:
    row = _developer_row(ignorable=True)
    encoded = _V4Codec().encode_event(row)
    assert encoded["type"] == "developer/message"
    assert encoded["ignorable"] is True
    assert encoded["data"]["message"]["id"] == "d1"


# ---------------------------------------------------------------------------
# Chunk 11: v1 -> v2 stream internals
# ---------------------------------------------------------------------------


def test_assert_chunk_envelope_branches() -> None:
    header = _header(1, "v1")
    chunk = {"type": "text-delta", "index": 0, "text": "a"}
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="has unexpected member bogus"
    ):
        _run_stage(
            SESSION_FORMAT_V1_TO_V2,
            header,
            0,
            [_chunk_row(0, 100, chunk, bogus=1)],
        )
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="lacks required member time"
    ):
        _run_stage(
            SESSION_FORMAT_V1_TO_V2,
            header,
            0,
            [
                {
                    "type": "assistant/chunk",
                    "seq": 0,
                    "data": {"turn": 1, "step": 1, "chunk": chunk},
                }
            ],
        )
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="ignorable must be true when present",
    ):
        _run_stage(
            SESSION_FORMAT_V1_TO_V2,
            header,
            0,
            [_chunk_row(0, 100, chunk, ignorable=False)],
        )


def test_assert_source_delivery_marker_v1_branches() -> None:
    marker = {
        "type": "session-log-deepseek/delivery-accepted",
        "seq": 1,
        "time": 101,
        "data": {"sessionId": "other", "throughSeq": 0, "sessionFormatVersion": 1},
    }
    turn_start = {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="names the wrong Session",
    ):
        _run_stage(SESSION_FORMAT_V1_TO_V2, _header(1, "v1"), 0, [turn_start, marker])
    # A marker before the inherited cut names a parent generation: tolerated.
    _, cut, values = _run_stage(
        SESSION_FORMAT_V1_TO_V2,
        _header(1, "v1", seedLength=2, parentSession="p"),
        2,
        [turn_start, marker],
    )
    assert values[1]["type"] == "session-log-deepseek/delivery-accepted"
    assert cut == 2


def test_transform_chunk_terminal_group() -> None:
    _, _, values = _run_stage(
        SESSION_FORMAT_V1_TO_V2,
        _header(1, "v1"),
        0,
        [
            _chunk_row(0, 100, {"type": "text-delta", "index": 0, "text": "a"}),
            _chunk_row(1, 101, {"type": "finish", "reason": {"kind": "stop"}}),
            {
                "type": "assistant/chunk",
                "seq": 2,
                "time": 102,
                "data": {
                    "turn": 2,
                    "step": 1,
                    "chunk": {"type": "text-delta", "index": 0, "text": "b"},
                },
            },
        ],
    )
    assert [event["type"] for event in values] == [
        "assistant/attempt",
        "assistant/attempt",
    ]
    assert values[0]["data"]["stream"][0]["texts"] == ["a"]
    assert values[1]["data"]["stream"][0]["texts"] == ["b"]


def test_transform_message_bare() -> None:
    message = {
        "type": "assistant/message",
        "seq": 0,
        "time": 100,
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "m",
                "role": "assistant",
                "content": [],
                "source": {"kind": "user"},
            },
        },
    }
    _, _, values = _run_stage(SESSION_FORMAT_V1_TO_V2, _header(1, "v1"), 0, [message])
    assert [event["type"] for event in values] == ["assistant/message"]
    assert values[0]["data"]["message"]["id"] == "m"


def test_transform_released_event_buffers_after_chunk() -> None:
    message = {
        "type": "assistant/message",
        "seq": 2,
        "time": 102,
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "m",
                "role": "assistant",
                "content": [],
                "source": {"kind": "user"},
            },
        },
        "sourceEventSeqs": [0],
    }
    _, _, values = _run_stage(
        SESSION_FORMAT_V1_TO_V2,
        _header(1, "v1"),
        0,
        [
            _chunk_row(0, 100, {"type": "text-delta", "index": 0, "text": "a"}),
            {
                "type": "user/message",
                "seq": 1,
                "time": 101,
                "data": {
                    "role": "user",
                    "id": "u1",
                    "content": [{"type": "text", "text": "hi"}],
                    "source": {"kind": "user"},
                },
            },
            message,
        ],
    )
    assert [event["type"] for event in values] == [
        "user/message",
        "assistant/message",
    ]
    assert values[1]["data"]["stream"][0]["texts"] == ["a"]


def test_matches_chunk_sources_direct() -> None:
    group = _AttemptGroup(1, 1)
    group.spans = [{"first_seq": 0, "event_count": 2}]
    group.chunk_count = 2
    assert _matches_chunk_sources(group, [0]) is False
    assert _matches_chunk_sources(group, [0, 5]) is False
    assert _matches_chunk_sources(group, [0, 1]) is True


def test_map_one_direct() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="targets consumed assistant/chunk 0",
    ):
        _map_one(0, {}, "x")
    assert _map_one(1, {1: 3}, "x") == 3


def test_remap_payload_references_direct() -> None:
    done = {
        "type": "command/done",
        "seq": 1,
        "data": {"commandId": "c", "kind": "run"},
    }
    assert _remap_payload_references(done, {0: 1}) == {
        "commandId": "c",
        "kind": "run",
    }
    summary = {
        "type": "compaction/summary",
        "seq": 1,
        "data": {
            "shadowedRange": {"start": 0, "end": 0},
            "shadowedSeqs": [0],
        },
    }
    remapped = _remap_payload_references(summary, {0: 1})
    assert remapped["shadowedRange"] == {"start": 1, "end": 1}
    assert remapped["shadowedSeqs"] == [1]


def test_remap_references_sources_via_stage() -> None:
    message = {
        "type": "assistant/message",
        "seq": 1,
        "time": 101,
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "m",
                "role": "assistant",
                "content": [],
                "source": {"kind": "user"},
            },
        },
        "sourceEventSeqs": [0],
    }
    _, _, values = _run_stage(
        SESSION_FORMAT_V1_TO_V2,
        _header(1, "v1"),
        0,
        [
            _chunk_row(0, 100, {"type": "text-delta", "index": 0, "text": "a"}),
            message,
            {
                "type": "user/message",
                "seq": 2,
                "time": 102,
                "data": {
                    "role": "user",
                    "id": "u2",
                    "content": [{"type": "text", "text": "next"}],
                    "source": {"kind": "user"},
                },
                "sourceEventSeqs": [1],
            },
        ],
    )
    assert values[1]["type"] == "user/message"
    # The citing message is the first emitted event (target seq 0), so the
    # source reference remaps 1 -> 0.
    assert values[1]["sourceEventSeqs"] == [0]


def test_map_one_stage_consumed_chunk() -> None:
    message = {
        "type": "assistant/message",
        "seq": 1,
        "time": 101,
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "m",
                "role": "assistant",
                "content": [],
                "source": {"kind": "user"},
            },
        },
        "sourceEventSeqs": [0],
    }
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="targets consumed assistant/chunk 0",
    ):
        _run_stage(
            SESSION_FORMAT_V1_TO_V2,
            _header(1, "v1"),
            0,
            [
                _chunk_row(0, 100, {"type": "text-delta", "index": 0, "text": "a"}),
                message,
                {
                    "type": "user/message",
                    "seq": 2,
                    "time": 102,
                    "data": {
                        "role": "user",
                        "id": "u2",
                        "content": [{"type": "text", "text": "next"}],
                        "source": {"kind": "user"},
                    },
                    "sourceEventSeqs": [0],
                },
            ],
        )


def test_packed_run_full_restore_and_straddle() -> None:
    packed = {
        "type": "text-chunks",
        "seq0": 1,
        "time0": 100,
        "data": {
            "turn": 1,
            "step": 1,
            "index": 0,
            "dt": [1],
            "texts": ["a", "b"],
        },
    }
    turn_start = {"type": "turn/start", "seq": 0, "time": 99, "data": {"turn": 1}}
    turn_end = {
        "type": "turn/end",
        "seq": 3,
        "time": 103,
        "data": {"turn": 1, "reason": {"kind": "completed"}},
    }
    artifact = _migrate(_header(0, "p0"), [turn_start, packed, turn_end])
    assert [event["type"] for event in artifact.events] == [
        "turn/start",
        "assistant/attempt",
        "turn/end",
    ]
    assert artifact.events[1]["data"]["stream"][0]["texts"] == ["a", "b"]
    # The same packed run straddling a seeded cut is refused at the chain.
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="splits one Assistant attempt",
    ):
        _migrate(
            _header(0, "p0", seedLength=2),
            [turn_start, packed, turn_end],
        )


def test_legacy_interrupted_turn_empty_inserted() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="does not close the prior turn",
    ):
        _run_stage(
            SESSION_FORMAT_V1_TO_V2,
            _header(1, "v1"),
            0,
            [
                {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}},
                {
                    "type": "step/start",
                    "seq": 1,
                    "time": 101,
                    "data": {"turn": 1, "step": 1},
                },
                {
                    "type": "step/end",
                    "seq": 2,
                    "time": 102,
                    "data": {"turn": 1, "step": 1},
                },
                {
                    "type": "agent/inbox/spliced",
                    "seq": 3,
                    "time": 103,
                    "data": {"target": "next-turn", "start": 3, "inserted": []},
                },
                {"type": "turn/start", "seq": 4, "time": 104, "data": {"turn": 2}},
            ],
        )


def test_finish_migration_v1_to_v2_seeded_synthesis() -> None:
    _, cut, values = _run_stage(
        SESSION_FORMAT_V1_TO_V2,
        _header(1, "v1", seedLength=2),
        2,
        [
            {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}},
            {
                "type": "turn/end",
                "seq": 1,
                "time": 101,
                "data": {"turn": 1, "reason": {"kind": "completed"}},
            },
        ],
    )
    assert cut == 2
    assert values[-1] == {
        "type": "session/end-seed",
        "seq": 2,
        "time": 101,
        "data": {"inherited": True},
    }


def test_transform_released_run_flushes_raw_chunk() -> None:
    stage, out = _v1_stage()
    stage.transform_event(_chunk_row(0, 100, {"type": "block-start", "index": 0}), out)
    stage.transform_run(_packed_run(1, 1, 1, ["a"]), out)
    assert stage.finish(out) == 0
    assert [event["type"] for event in out.values] == ["assistant/attempt"]
    assert out.values[0]["data"]["stream"][0]["type"] == "chunk"
    assert out.values[0]["data"]["stream"][1]["texts"] == ["a"]


# ---------------------------------------------------------------------------
# Chunk 11: v2 -> v3 stage internals
# ---------------------------------------------------------------------------


def test_observe_message_ids_collision() -> None:
    system_id = _system_id("v2", 1, "step/start")
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="collides with a generated system message id",
    ):
        _run_stage(
            SESSION_FORMAT_V2_TO_V3,
            V2_HEADER,
            0,
            [*_v2_opening(), _v2_user(2, 1002, system_id, "hi")],
        )


def test_emit_system_collision_with_observed_id() -> None:
    # A title request is not a surface event, so it can be observed before
    # the first step; its message id then collides with the generated head.
    system_id = _system_id("v2", 2, "step/start")
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="collides with an existing message id",
    ):
        _run_stage(
            SESSION_FORMAT_V2_TO_V3,
            V2_HEADER,
            0,
            [
                {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
                {
                    "type": "session/title-llm-request",
                    "seq": 1,
                    "time": 1001,
                    "data": {
                        "titleProvider": "p",
                        "messageSeqs": [],
                        "route": "r",
                        "system": "s",
                        "messages": [
                            {
                                "id": system_id,
                                "role": "user",
                                "content": [{"type": "text", "text": "hi"}],
                                "source": {"kind": "user"},
                            }
                        ],
                        "maxTokens": 10,
                    },
                },
                {
                    "type": "step/start",
                    "seq": 2,
                    "time": 1002,
                    "data": {"turn": 1, "step": 1},
                },
            ],
        )


def test_v2_to_v3_delivery_marker_branches() -> None:
    opening = _v2_opening()
    with pytest.raises(SessionFormatError, match="claims target format v3"):
        _run_stage(
            SESSION_FORMAT_V2_TO_V3,
            V2_HEADER,
            0,
            [
                *opening,
                {
                    "type": "session-log-deepseek/delivery-accepted",
                    "seq": 2,
                    "time": 1002,
                    "data": {
                        "sessionId": "v2",
                        "throughSeq": 0,
                        "sessionFormatVersion": 3,
                    },
                },
            ],
        )
    # A foreign current-generation marker is recorded at transform time and
    # refused at finish when the session has no parent to own it.
    with pytest.raises(SessionFormatError, match="names the wrong Session"):
        _run_stage(
            SESSION_FORMAT_V2_TO_V3,
            _header(2, "v2", isSeeded=True),
            5,
            [
                *opening,
                {
                    "type": "session-log-deepseek/delivery-accepted",
                    "seq": 2,
                    "time": 1002,
                    "data": {
                        "sessionId": "other",
                        "throughSeq": 0,
                        "sessionFormatVersion": 2,
                    },
                },
                {
                    "type": "step/end",
                    "seq": 3,
                    "time": 1003,
                    "data": {"turn": 1, "step": 1},
                },
                {
                    "type": "turn/end",
                    "seq": 4,
                    "time": 1004,
                    "data": {"turn": 1, "reason": {"kind": "completed"}},
                },
                {
                    "type": "session/end-seed",
                    "seq": 5,
                    "time": 1005,
                    "data": {"inherited": True},
                },
            ],
        )


def test_canonicalize_transformed_event_branches() -> None:
    with pytest.raises(
        SessionFormatError, match="requires exact replace fields op/start/end"
    ):
        _canonicalize_transformed_event(
            {
                "type": "user/message",
                "seq": 2,
                "time": 100,
                "data": {
                    "role": "user",
                    "id": "m1",
                    "content": [],
                    "source": {"kind": "user"},
                },
                "surfaceOp": {"op": "move", "start": 0, "end": 0},
            }
        )
    # Empty tools / adapterDefaults are dropped from a request/header.
    _, _, values = _run_stage(
        SESSION_FORMAT_V2_TO_V3,
        V2_HEADER,
        0,
        [
            *_v2_opening(),
            {
                "type": "request/header",
                "seq": 2,
                "time": 1002,
                "data": {
                    "header": {"tools": [], "adapterDefaults": {}},
                    "reason": {"kind": "start"},
                },
            },
        ],
    )
    assert values[3]["type"] == "request/header"
    assert values[3]["data"]["header"] == {}


def test_observe_message_ids_branches() -> None:
    logical = {
        "version": 2,
        "id": "v2",
        "createdAt": 1000,
        "isSeeded": False,
        "delegationDepth": 0,
    }
    stage = SESSION_FORMAT_V2_TO_V3.create_stage(
        SessionFormatMigrationStageInput(
            source_header=dict(logical),
            target_header=SESSION_FORMAT_V2_TO_V3.migrate_header(dict(logical)),
            source_inherited_event_count=0,
            source_kind="decoded",
        )
    )
    stage._observe_message_ids(
        {
            "type": "agent/inbox/spliced",
            "seq": 0,
            "time": 100,
            "data": {"target": "next-turn", "start": 0, "inserted": [{"id": "m1"}]},
        }
    )
    stage._observe_message_ids(
        {
            "type": "session/title-llm-request",
            "seq": 1,
            "time": 101,
            "data": {"messages": [{"id": "m2"}]},
        }
    )
    # A non-object member of an inserted list is skipped, not rejected.
    stage._observe_message_ids(
        {
            "type": "agent/inbox/spliced",
            "seq": 2,
            "time": 102,
            "data": {
                "target": "next-turn",
                "start": 2,
                "inserted": ["x", {"id": "m3"}],
            },
        }
    )
    assert stage._original_ids == {"m1", "m2", "m3"}


# ---------------------------------------------------------------------------
# Chunk 11: v3 -> v4 source rewriting and child facts
# ---------------------------------------------------------------------------


def test_producer_kind_and_plugin_source_rewrite() -> None:
    assert _producer_kind("goal", "user") == "goal"
    with pytest.raises(SessionFormatError, match="is not canonical"):
        _rewrite_plugin_source({"kind": "plugin", "plugin": 3}, 0, "user")
    assert _rewrite_plugin_source(
        {"kind": "plugin", "plugin": "goal", "extra": 1}, 0, "user"
    ) == {"kind": "goal", "extra": 1}
    with pytest.raises(SessionFormatError, match="requires a nonempty kind"):
        rewrite_v3_message_source({}, 0, "user")
    assert rewrite_v3_message_source({"kind": "user"}, 0, "user") == {"kind": "user"}
    message = {"id": "m", "role": "user", "content": []}
    assert _rewrite_event_message_source(message, 0) is message


def test_lift_tool_result_branches() -> None:
    turn_start = {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}
    assert lift_tool_result(turn_start) is turn_start
    with pytest.raises(SessionFormatError, match="isError must be boolean"):
        lift_tool_result(
            _v3_tool_result(
                0,
                100,
                content=[
                    {
                        "type": "tool-result",
                        "toolCallId": "c1",
                        "content": [{"type": "text", "text": "3"}],
                        "isError": "yes",
                    }
                ],
            )
        )
    with pytest.raises(
        SessionFormatError, match="tool-result content must be an array"
    ):
        _result_content("x", "subject")


def test_migrate_v3_block_and_content() -> None:
    with pytest.raises(
        SessionFormatError, match="requires content blocks with string type tags"
    ):
        _migrate_v3_block("x", "s")
    with pytest.raises(SessionFormatError, match="content must be an array"):
        _migrate_v3_content("x", "s")
    assert _migrate_v3_block({"type": "widget", "x": 1}, "s") == {
        "type": "plugin:widget",
        "x": 1,
    }
    assert _migrate_v3_block({"type": "text", "text": "hi"}, "s") == {
        "type": "text",
        "text": "hi",
    }


def test_historical_child_catalog_source_branches() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="requires a subagent child with a direct parent",
    ):
        historical_child_catalog_source(_artifact(3, []))
    header = {
        "version": 3,
        "id": "c1",
        "createdAt": 500,
        "isSeeded": False,
        "delegationDepth": 0,
        "origin": "subagent",
        "parentSession": "p",
    }
    descriptor = {
        "type": "subagent/descriptor",
        "seq": 0,
        "time": 100,
        "data": {"version": 2, "provider": "p", "mode": "one-shot", "label": "L"},
    }
    source = historical_child_catalog_source(
        SessionFormatArtifact(
            header=header, events=[descriptor], inherited_event_count=0
        )
    )
    assert source == {
        "childId": "c1",
        "childCreatedAt": 500,
        "descriptorCount": 1,
        "descriptor": descriptor["data"],
    }
    empty = historical_child_catalog_source(
        SessionFormatArtifact(header=header, events=[], inherited_event_count=0)
    )
    assert empty["descriptorCount"] == 0
    assert empty["descriptor"] is None


def test_child_catalog_fact_provider_and_subject() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match=r"\(raw log: sessions/c1\.jsonl\)",
    ):
        child_catalog_fact(
            {
                "childId": "c1",
                "childCreatedAt": 500,
                "descriptorCount": 1,
                "descriptor": {"version": 2, "provider": 3},
                "sourcePath": "sessions/c1.jsonl",
            }
        )
    assert child_catalog_subject({"childId": "c1"}) == "Session c1"
    assert (
        child_catalog_subject({"childId": "c1", "sourcePath": "sessions/c1.jsonl"})
        == "Session c1 (raw log: sessions/c1.jsonl)"
    )


def _v3_seeded_rows() -> list[dict]:
    return [
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}},
        {"type": "step/start", "seq": 1, "time": 101, "data": {"turn": 1, "step": 1}},
        {
            "type": "step/end",
            "seq": 2,
            "time": 102,
            "data": {"turn": 1, "step": 1},
        },
        {
            "type": "turn/end",
            "seq": 3,
            "time": 103,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
        {
            "type": "session/end-seed",
            "seq": 4,
            "time": 104,
            "data": {"inherited": True},
        },
    ]


def test_v3_to_v4_stage_dense_and_unseeded_marker() -> None:
    with pytest.raises(SessionFormatError, match="V3 source events must be dense"):
        _run_stage(
            _v3_migration([]),
            V3_HEADER,
            0,
            [
                {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}},
                {
                    "type": "step/start",
                    "seq": 2,
                    "time": 102,
                    "data": {"turn": 1, "step": 1},
                },
            ],
            kind="transformed",
        )
    with pytest.raises(
        SessionFormatError,
        match="unseeded format v3 Session contains an inherited end-seed marker",
    ):
        _run_stage(
            _v3_migration([]),
            V3_HEADER,
            0,
            [
                {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}},
                {
                    "type": "session/end-seed",
                    "seq": 1,
                    "time": 101,
                    "data": {"inherited": True},
                },
            ],
            kind="transformed",
        )


def test_v3_to_v4_stage_foreign_delivery_and_cut() -> None:
    delivery = {
        "type": "session-log-deepseek/delivery-accepted",
        "seq": 1,
        "time": 101,
        "data": {"sessionId": "other", "throughSeq": 0, "sessionFormatVersion": 3},
    }
    rows = _v3_seeded_rows()
    rows.insert(1, delivery)
    for index, row in enumerate(rows):
        row["seq"] = index
        row["time"] = 100 + index
    with pytest.raises(SessionFormatError, match="names the wrong Session"):
        _run_stage(
            _v3_migration([]),
            _header(3, "v3", isSeeded=True),
            5,
            rows,
            kind="transformed",
        )
    # The marker's source cut disagrees with the declared cut.
    with pytest.raises(
        SessionFormatError, match="inherited cut disagrees with its source marker"
    ):
        _run_stage(
            _v3_migration([]),
            _header(3, "v3", isSeeded=True),
            1,
            _v3_seeded_rows(),
            kind="transformed",
        )


def _v3_child_source() -> dict:
    return {
        "childId": "c1",
        "childCreatedAt": 500,
        "descriptorCount": 1,
        "descriptor": {"version": 2, "provider": "p", "mode": "one-shot", "label": "L"},
    }


def _v3_catalog_rows(catalog_data: dict) -> list[dict]:
    # The catalog sits after the end-seed marker: catalogs before the cut
    # belong to the inherited prefix and are reset by the marker.
    rows = [
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}},
        {"type": "step/start", "seq": 1, "time": 101, "data": {"turn": 1, "step": 1}},
        {"type": "step/end", "seq": 2, "time": 102, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": 3,
            "time": 103,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
        {
            "type": "session/end-seed",
            "seq": 4,
            "time": 104,
            "data": {"inherited": True},
        },
        {
            "type": "subagent/catalog",
            "seq": 5,
            "time": 105,
            "data": catalog_data,
        },
    ]
    return rows


def test_v3_to_v4_stage_catalog_conflict_and_match() -> None:
    conflicting = {
        "version": 0,
        "childId": "c1",
        "childCreatedAt": 500,
        "mode": "continuable",
        "label": "L2",
    }
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="conflicts with its parent catalog",
    ):
        _run_stage(
            _v3_migration([_v3_child_source()]),
            _header(3, "v3", isSeeded=True),
            4,
            _v3_catalog_rows(conflicting),
            kind="transformed",
        )
    matching = {
        "version": 0,
        "childId": "c1",
        "childCreatedAt": 500,
        "mode": "one-shot",
        "label": "L",
    }
    _, cut, values = _run_stage(
        _v3_migration([_v3_child_source()]),
        _header(3, "v3", isSeeded=True),
        4,
        _v3_catalog_rows(matching),
        kind="transformed",
    )
    assert cut == 4
    catalogs = [event for event in values if event["type"] == "subagent/catalog"]
    assert len(catalogs) == 1
    assert catalogs[0]["seq"] == 5


def test_v3_to_v4_stage_transform_run() -> None:
    migration = _v3_migration([])
    logical = {
        "version": 3,
        "id": "v3",
        "createdAt": 1000,
        "isSeeded": False,
        "delegationDepth": 0,
    }
    stage = migration.create_stage(
        SessionFormatMigrationStageInput(
            source_header=dict(logical),
            target_header=migration.migrate_header(dict(logical)),
            source_inherited_event_count=0,
            source_kind="transformed",
        )
    )
    out = _Collector()
    stage.transform_run(
        SessionFormatEventRun(
            run_type="plain",
            first_seq=0,
            event_count=1,
            expand_events=[
                {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}
            ],
        ),
        out,
    )
    assert [event["type"] for event in out.values] == ["turn/start"]


# ---------------------------------------------------------------------------
# Chunk 11: chain / catalog / restore-object paths
# ---------------------------------------------------------------------------


def test_stage_base_transform_run_and_finish() -> None:
    assert _Stage().finish(_Collector()) == 0
    logical = {
        "version": 0,
        "id": "v0",
        "createdAt": 1000,
        "isSeeded": False,
        "delegationDepth": 0,
    }
    stage = SESSION_FORMAT_V0_TO_V1.create_stage(
        SessionFormatMigrationStageInput(
            source_header=dict(logical),
            target_header=SESSION_FORMAT_V0_TO_V1.migrate_header(dict(logical)),
            source_inherited_event_count=0,
            source_kind="decoded",
        )
    )
    out = _Collector()
    stage.transform_run(
        SessionFormatEventRun(
            run_type="plain",
            first_seq=0,
            event_count=1,
            expand_events=[
                {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}
            ],
        ),
        out,
    )
    assert [event["type"] for event in out.values] == ["turn/start"]


def test_define_session_format_migration_rejections() -> None:
    def _bad_migration(**overrides: object) -> SessionFormatMigration:
        fields: dict[str, Any] = {
            "name": "bad-migration",
            "from_version": 0,
            "to_version": 1,
            "migrate_header": SESSION_FORMAT_V0_TO_V1.migrate_header,
            "create_stage": lambda input_: _Stage(),
            "validate_target_header": assert_v1_header,
        }
        fields.update(overrides)
        return SessionFormatMigration(**fields)

    with pytest.raises(SessionFormatError, match="name must be a non-empty string"):
        define_session_format_migration(_bad_migration(name=""))
    with pytest.raises(SessionFormatError, match="must declare adjacent v0->v1"):
        define_session_format_migration(_bad_migration(to_version=2))


def test_refusal_wraps_plain_error() -> None:
    wrapped = _refusal(SESSION_FORMAT_V0_TO_V1, SessionFormatError("boom"))
    assert isinstance(wrapped, SessionFormatUnsupportedMigrationError)
    assert str(wrapped) == (
        "session-format-v0-to-v1 refuses this format v0 Session: boom"
    )
    original = SessionFormatUnsupportedMigrationError("already")
    assert _refusal(SESSION_FORMAT_V0_TO_V1, original) is original


def test_assert_header_version_direct() -> None:
    with pytest.raises(SessionFormatError, match="expected format v1 header"):
        _assert_header_version({"version": 0}, 1)
    _assert_header_version({"version": 1}, 1)


def _shipped_migrations() -> list[SessionFormatMigration]:
    return [
        SESSION_FORMAT_V0_TO_V1,
        SESSION_FORMAT_V1_TO_V2,
        SESSION_FORMAT_V2_TO_V3,
        SESSION_FORMAT_V3_TO_V4,
    ]


def test_chain_construction_rejections() -> None:
    with pytest.raises(SessionFormatError, match="is duplicated"):
        SessionFormatChain(
            4,
            [SESSION_FORMAT_V0_TO_V1, SESSION_FORMAT_V0_TO_V1],
            assert_v4_header,
        )
    extra = SessionFormatMigration(
        name="session-format-v4-to-v5",
        from_version=4,
        to_version=5,
        migrate_header=lambda header: {**header, "version": 5},
        create_stage=lambda input_: _Stage(),
        validate_target_header=lambda header: header,
    )
    with pytest.raises(SessionFormatError, match="does not lead to current v4"):
        SessionFormatChain(4, [*_shipped_migrations(), extra], assert_v4_header)
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="uses newer format v9"
    ):
        SESSION_FORMAT_CATALOG._chain._plan(9)


def test_chain_migrate_header_bad_restorer() -> None:
    chain = SessionFormatChain(
        4,
        _shipped_migrations(),
        lambda header: {**header, "version": 3},
    )
    with pytest.raises(SessionFormatError, match="restorer returned v3; expected v4"):
        chain.migrate_header(
            {
                "version": 0,
                "id": "s",
                "createdAt": 1000,
                "isSeeded": False,
                "delegationDepth": 0,
            }
        )


def test_advance_header_branches() -> None:
    chain = SessionFormatChain(4, _shipped_migrations(), assert_v4_header)
    logical = {
        "version": 0,
        "id": "s",
        "createdAt": 1000,
        "isSeeded": False,
        "delegationDepth": 0,
    }

    def _boom(header: dict) -> dict:
        raise SessionFormatError("boom")

    bad_migrate = SessionFormatMigration(
        name="session-format-v0-to-v1",
        from_version=0,
        to_version=1,
        migrate_header=_boom,
        create_stage=lambda input_: _Stage(),
        validate_target_header=assert_v1_header,
    )
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="refuses this format v0 Session header: boom",
    ):
        chain._advance_header(bad_migrate, logical)
    wrong_version = SessionFormatMigration(
        name="session-format-v0-to-v1",
        from_version=0,
        to_version=1,
        migrate_header=lambda header: {**header, "version": 3},
        create_stage=lambda input_: _Stage(),
        validate_target_header=assert_v1_header,
    )
    with pytest.raises(SessionFormatError, match="header returned v3; expected v1"):
        chain._advance_header(wrong_version, logical)

    def _bad_target(header: Any) -> dict:
        raise SessionFormatError("bad target")

    bad_validate = SessionFormatMigration(
        name="session-format-v0-to-v1",
        from_version=0,
        to_version=1,
        migrate_header=SESSION_FORMAT_V0_TO_V1.migrate_header,
        create_stage=lambda input_: _Stage(),
        validate_target_header=_bad_target,
    )
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="refuses this format v0 Session header: bad target",
    ):
        chain._advance_header(bad_validate, logical)


def test_create_stream_bare_edge() -> None:
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="requires explicit historical child facts",
    ):
        SESSION_FORMAT_CATALOG._chain.create_stream(
            {
                "version": 3,
                "id": "v3",
                "createdAt": 1000,
                "isSeeded": False,
                "delegationDepth": 0,
            },
            0,
            _Collector(),
        )


def test_chained_context_finish_branches() -> None:
    class _FailingStage(_Stage):
        def finish(self, emit: _Emit) -> int:
            raise SessionFormatError("boom")

    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="refuses this format v0 Session: boom",
    ):
        _ChainedContext(SESSION_FORMAT_V0_TO_V1, _FailingStage(), _Collector()).finish()

    class _CutStage(_Stage):
        header_inherited_event_count: int | None = 2

        def finish(self, emit: _Emit) -> int:
            return 5

    with pytest.raises(
        SessionFormatError, match="changed its predeclared inherited cut"
    ):
        _ChainedContext(SESSION_FORMAT_V0_TO_V1, _CutStage(), _Collector()).finish()


def test_finish_decoder_and_restore_current_version_mismatch() -> None:
    with pytest.raises(
        SessionFormatError, match="changed its predeclared inherited cut"
    ):
        _finish_decoder(_v0_decoder("strict"), 5)
    with pytest.raises(SessionFormatError, match="restorer returned v3; expected v4"):
        _restore_current_version(_artifact(3, []), 4)


def test_create_restore_current_validation() -> None:
    turn_start = {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}
    restore = SESSION_FORMAT_CATALOG.create_restore(
        _header(4, "cur"), validation="current"
    )
    restore.decode_row(turn_start)
    artifact = restore.finish()
    assert artifact.header["version"] == 4
    assert [event["type"] for event in artifact.events] == ["turn/start"]
    restore = HISTORICAL_SESSION_FORMAT_CATALOG.create_restore(
        _header(3, "h"), validation="current"
    )
    restore.decode_row(turn_start)
    artifact = restore.finish()
    assert artifact.header["version"] == 3


def test_migrating_restore_finish_rewraps_plain_error() -> None:
    catalog = create_session_format_catalog_with_children([])
    decoder = _V01Codec(0).create_decoder(_header(0, "s"), "strict")
    collector = _Collector()
    migration = catalog._chain.create_stream(
        decoder.header, decoder.header_inherited_event_count, collector
    )

    def _boom(artifact: SessionFormatArtifact) -> SessionFormatArtifact:
        raise SessionFormatError("boom")

    restore = _MigratingSessionFormatRestore(
        decoder, 0, migration, collector, _boom, "transformed", 0, 4
    )
    restore.decode_row(
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}
    )
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="refuses the transformed artifact: boom",
    ):
        restore.finish()


def _custom_catalog(current_encoder: Any) -> SessionFormatCatalog:
    return create_session_format_catalog(
        current_version=4,
        codecs=[_V01Codec(0), _V01Codec(1), _V2Codec(), _V3Codec(), _V4Codec()],
        current_encoder=current_encoder,
        migrations=_shipped_migrations(),
        restore_current=_restore_current_v4,
        restore_transformed_current=_restore_current_v4,
        restore_current_header=assert_v4_header,
    )


def test_read_header_unsupported_migration_branch() -> None:
    def _bad_target(header: Any) -> dict:
        raise SessionFormatError("bad target")

    bad_v3_to_v4 = SessionFormatMigration(
        name=SESSION_FORMAT_V3_TO_V4.name,
        from_version=3,
        to_version=4,
        migrate_header=SESSION_FORMAT_V3_TO_V4.migrate_header,
        create_stage=SESSION_FORMAT_V3_TO_V4.create_stage,
        validate_target_header=_bad_target,
    )
    catalog = create_session_format_catalog(
        current_version=4,
        codecs=[_V01Codec(0), _V01Codec(1), _V2Codec(), _V3Codec(), _V4Codec()],
        current_encoder=_V4Codec(),
        migrations=[
            SESSION_FORMAT_V0_TO_V1,
            SESSION_FORMAT_V1_TO_V2,
            SESSION_FORMAT_V2_TO_V3,
            bad_v3_to_v4,
        ],
        restore_current=_restore_current_v4,
        restore_transformed_current=_restore_current_v4,
        restore_current_header=assert_v4_header,
    )
    result = catalog.read_header(_header(3, "s"))
    assert result.status == "unsupported"
    assert result.reason is not None
    assert "bad target" in result.reason


def test_encode_current_header_wrong_encoder_version() -> None:
    class _WrongVersionEncoder:
        """Test double: accepts a v4 header but returns a v3 one."""

        def encode_header(self, header: dict, inherited_event_count: int) -> dict:
            return {**header, "version": 3}

        def encode_event(self, event: dict) -> dict:
            return event

    catalog = _custom_catalog(_WrongVersionEncoder())
    with pytest.raises(
        SessionFormatError,
        match="current Session codec returned a non-current header",
    ):
        catalog.encode_current_header(
            {
                "version": 4,
                "id": "s",
                "createdAt": 1000,
                "isSeeded": False,
                "delegationDepth": 0,
            },
            0,
        )


def test_v3_decoder_finish_issue_branches() -> None:
    bad_user = {
        "type": "user/message",
        "seq": 0,
        "time": 100,
        "data": {
            "role": "user",
            "id": "m",
            "content": [],
            "source": {"kind": "user"},
        },
    }
    decoder = _V3Codec().create_decoder(_header(3, "s", isSeeded=True), "recoverable")
    decoder.decode_row(bad_user, _Collector())
    with pytest.raises(
        SessionFormatError,
        match="lacks an accepted inherited end-seed marker",
    ):
        decoder.finish()
    decoder = _V3Codec().create_decoder(_header(3, "s"), "recoverable")
    decoder.decode_row(
        {
            "type": "session/end-seed",
            "seq": 0,
            "time": 100,
            "data": {"inherited": True},
        },
        _Collector(),
    )
    decoder.decode_row({**bad_user, "seq": 1}, _Collector())
    with pytest.raises(
        SessionFormatError,
        match="contains an inherited end-seed marker",
    ):
        decoder.finish()
    decoder = _V3Codec().create_decoder(_header(3, "s"), "recoverable")
    decoder.decode_row(bad_user, _Collector())
    assert decoder.finish() == 0


def test_v2_decoder_recoverable_gap_return() -> None:
    decoder = _v2_decoder("recoverable")
    out = _Collector()
    decoder.decode_row(
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}, out
    )
    decoder.decode_row(
        {"type": "turn/start", "seq": 2, "time": 101, "data": {"turn": 1}}, out
    )
    assert [event["seq"] for event in out.values] == [0]
    with pytest.raises(SessionFormatError, match="seq gap"):
        decoder.decode_row(
            {
                "type": "turn/end",
                "seq": 3,
                "time": 102,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
            out,
        )


def test_normalize_legacy_compaction_present_invalid_id() -> None:
    state = _LegacyCompactionState()
    event = {
        "type": "compaction/start",
        "seq": 0,
        "time": 100,
        "data": {"compactionId": "", "turn": 1},
    }
    assert _normalize_legacy_compaction(event, "s", state) is event
    assert state.compaction_id is None


# ---------------------------------------------------------------------------
# Chunk 12: remaining decoder / admission / stage branches
# ---------------------------------------------------------------------------


def test_decoder_second_recoverable_error_keeps_first_issue() -> None:
    decoder = _v0_decoder("recoverable")
    out = _Collector()
    decoder.decode_row("not an object", out)
    decoder.decode_row("also not an object", out)
    assert out.values == []
    with pytest.raises(SessionFormatError, match="must be a JSON object"):
        decoder.decode_row(
            {
                "type": "turn/end",
                "seq": 0,
                "time": 100,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
            out,
        )
    decoder = _v2_decoder("recoverable")
    out = _Collector()
    decoder.decode_row("not an object", out)
    decoder.decode_row("also not an object", out)
    assert out.values == []
    with pytest.raises(SessionFormatError, match="must be a JSON object"):
        decoder.decode_row(
            {
                "type": "turn/end",
                "seq": 0,
                "time": 100,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
            out,
        )


def test_v2_decoder_end_seed_without_inherited() -> None:
    decoder = _v2_decoder("strict")
    out = _Collector()
    decoder.decode_row(
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}, out
    )
    decoder.decode_row(
        {"type": "session/end-seed", "seq": 1, "time": 101, "data": {}}, out
    )
    assert [event["type"] for event in out.values] == [
        "turn/start",
        "session/end-seed",
    ]
    assert decoder.finish() == 0


def test_v4_developer_data_fallthrough_branches() -> None:
    assert_v4_developer_data({"type": "user/message", "seq": 0, "time": 1, "data": "x"})
    assert_v4_developer_data(
        {
            "type": "agent/inbox/spliced",
            "seq": 0,
            "time": 100,
            "data": {"target": "next-turn", "start": 0, "inserted": "x"},
        }
    )
    assert_v4_developer_data(
        {
            "type": "agent/inbox/spliced",
            "seq": 0,
            "time": 100,
            "data": {
                "target": "next-turn",
                "start": 0,
                "inserted": [
                    "x",
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                ],
            },
        }
    )
    # Stream entries whose chunk is not an object, and block-start chunks
    # that are not tool changes, are skipped.
    assert_v4_developer_data(
        {
            "type": "assistant/message",
            "seq": 0,
            "time": 100,
            "data": {"stream": [{"type": "chunk", "chunk": "x"}]},
        }
    )
    assert_v4_developer_data(
        {
            "type": "assistant/message",
            "seq": 0,
            "time": 100,
            "data": {
                "stream": [
                    {
                        "type": "chunk",
                        "chunk": {"type": "block-start", "blockType": "text"},
                    }
                ]
            },
        }
    )


def test_v4_retired_syntax_spliced_and_stream_branches() -> None:
    assert_v4_retired_syntax(
        {
            "type": "agent/inbox/spliced",
            "seq": 0,
            "time": 100,
            "data": {"target": "next-turn", "start": 0, "inserted": "x"},
        }
    )
    assert_v4_retired_syntax(
        {
            "type": "agent/inbox/spliced",
            "seq": 0,
            "time": 100,
            "data": {
                "target": "next-turn",
                "start": 0,
                "inserted": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]}
                ],
            },
        }
    )
    assert_v4_retired_syntax(
        {
            "type": "assistant/message",
            "seq": 0,
            "time": 100,
            "data": {
                "stream": [
                    {
                        "type": "chunk",
                        "chunk": {"type": "block-start", "blockType": "text"},
                    }
                ]
            },
        }
    )


def test_v4_system_message_fields_valid_blocks() -> None:
    def _system_data(content: list) -> dict:
        return {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "sys1",
                "role": "system",
                "content": content,
                "source": {"kind": "system-prompt"},
            },
        }

    assert_v4_system_message_fields(
        _system_row(
            data=_system_data(
                [
                    {
                        "type": "tool-call",
                        "id": "c1",
                        "name": "add",
                        "arguments": "{}",
                    }
                ]
            )
        )
    )
    assert_v4_system_message_fields(
        _system_row(
            data=_system_data(
                [
                    {
                        "type": "image",
                        "attachment": {
                            "attachmentId": "a",
                            "mediaType": "image/png",
                            "bytes": 10,
                            "width": 1,
                            "height": 2,
                            "name": "n",
                            "originalDimensions": {"width": 3, "height": 4},
                        },
                    }
                ]
            )
        )
    )
    assert_v4_system_message_fields(
        _system_row(data=_system_data([{"type": "file", "attachmentId": "a"}]))
    )


def test_v0_delivery_correct_session_and_chunk_passthrough() -> None:
    _, _, values = _run_stage(
        SESSION_FORMAT_V0_TO_V1,
        V0_HEADER,
        0,
        [
            {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}},
            {
                "type": "session-log-deepseek/delivery-accepted",
                "seq": 1,
                "time": 101,
                "data": {"sessionId": "v0", "throughSeq": 0},
            },
        ],
    )
    assert values[1]["type"] == "session-log-deepseek/delivery-accepted"
    _, _, values = _run_stage(
        SESSION_FORMAT_V0_TO_V1,
        V0_HEADER,
        0,
        [_chunk_row(0, 100, {"type": "text-delta", "index": 0, "text": "a"})],
    )
    assert [event["type"] for event in values] == ["assistant/chunk"]


def test_remap_payload_command_done_with_source_via_stage() -> None:
    message = {
        "type": "assistant/message",
        "seq": 1,
        "time": 101,
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "m",
                "role": "assistant",
                "content": [],
                "source": {"kind": "user"},
            },
        },
        "sourceEventSeqs": [0],
    }
    _, _, values = _run_stage(
        SESSION_FORMAT_V1_TO_V2,
        _header(1, "v1"),
        0,
        [
            _chunk_row(0, 100, {"type": "text-delta", "index": 0, "text": "a"}),
            message,
            {
                "type": "command/done",
                "seq": 2,
                "time": 102,
                "data": {"commandId": "c", "kind": "run", "sourceEventSeq": 1},
            },
        ],
    )
    assert values[1]["type"] == "command/done"
    assert values[1]["data"]["sourceEventSeq"] == 0


def test_assert_owned_content_raw_output_branches() -> None:
    base: dict[str, Any] = {
        "compactionId": "c1",
        "summary": [{"type": "text", "text": "s"}],
        "shadowedRange": {"start": 0, "end": 0},
        "shadowedSeqs": [0],
        "shadowedTokenCount": 1,
        "provider": "p",
        "model": "m",
    }
    with pytest.raises(SessionFormatError, match="widget"):
        _assert_event_v2(
            {
                "type": "compaction/summary",
                "seq": 0,
                "time": 100,
                "data": {**base, "rawOutput": [{"type": "widget"}]},
            }
        )
    _assert_event_v2(
        {"type": "compaction/summary", "seq": 0, "time": 100, "data": base}
    )


def test_remap_command_done_without_source_event_seq() -> None:
    event = {
        "type": "command/done",
        "seq": 1,
        "time": 100,
        "data": {"commandId": "c", "kind": "run"},
    }
    assert _remap_event_v2_to_v3(event, 5, [0, 1])["seq"] == 5
    assert _remap_event_v2_to_v3(event, 5, [0, 1])["data"] == {
        "commandId": "c",
        "kind": "run",
    }
    assert _remap_v3_references(event, 5, [0, 1])["seq"] == 5
    assert _remap_v3_references(event, 5, [0, 1])["data"] == {
        "commandId": "c",
        "kind": "run",
    }


def test_v2_to_v3_stage_unseeded_marker_delivery_and_second_step() -> None:
    with pytest.raises(
        SessionFormatError,
        match="unseeded Session contains an inherited end-seed marker",
    ):
        _run_stage(
            SESSION_FORMAT_V2_TO_V3,
            V2_HEADER,
            0,
            [
                *_v2_opening(),
                {
                    "type": "session/end-seed",
                    "seq": 2,
                    "time": 1002,
                    "data": {"inherited": True},
                },
            ],
        )
    _, cut, values = _run_stage(
        SESSION_FORMAT_V2_TO_V3,
        V2_HEADER,
        0,
        [
            *_v2_opening(),
            {
                "type": "session-log-deepseek/delivery-accepted",
                "seq": 2,
                "time": 1002,
                "data": {
                    "sessionId": "v2",
                    "throughSeq": 0,
                    "sessionFormatVersion": 2,
                },
            },
            {
                "type": "step/end",
                "seq": 3,
                "time": 1003,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "turn/end",
                "seq": 4,
                "time": 1004,
                "data": {"turn": 1, "reason": {"kind": "completed"}},
            },
        ],
    )
    assert cut == 0
    assert values[3]["type"] == "session-log-deepseek/delivery-accepted"
    _, _, values = _run_stage(
        SESSION_FORMAT_V2_TO_V3,
        V2_HEADER,
        0,
        [
            *_v2_opening(),
            {
                "type": "step/end",
                "seq": 2,
                "time": 1002,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "step/start",
                "seq": 3,
                "time": 1003,
                "data": {"turn": 1, "step": 2},
            },
        ],
    )
    assert [event["type"] for event in values] == [
        "turn/start",
        "step/start",
        "system/message",
        "step/end",
        "step/start",
    ]
    with pytest.raises(
        SessionFormatError,
        match="disagrees with its source cut",
    ):
        _run_stage(
            SESSION_FORMAT_V2_TO_V3,
            _header(2, "v2", isSeeded=True),
            2,
            [
                *_v2_opening(),
                _v2_user(2, 1002, "m1", "hi"),
                {
                    "type": "session/end-seed",
                    "seq": 3,
                    "time": 1003,
                    "data": {"inherited": True},
                },
            ],
        )


def test_v3_to_v4_stage_delivery_correct_session() -> None:
    _, cut, values = _run_stage(
        _v3_migration([]),
        V3_HEADER,
        0,
        [
            {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}},
            {
                "type": "session-log-deepseek/delivery-accepted",
                "seq": 1,
                "time": 101,
                "data": {
                    "sessionId": "v3",
                    "throughSeq": 0,
                    "sessionFormatVersion": 3,
                },
            },
            {
                "type": "step/start",
                "seq": 2,
                "time": 102,
                "data": {"turn": 1, "step": 1},
            },
            {"type": "step/end", "seq": 3, "time": 103, "data": {"turn": 1, "step": 1}},
            {
                "type": "turn/end",
                "seq": 4,
                "time": 104,
                "data": {"turn": 1, "reason": {"kind": "completed"}},
            },
        ],
    )
    assert cut == 0
    assert values[1]["type"] == "session-log-deepseek/delivery-accepted"


def test_v3_to_v4_stage_catalog_created_at_conflict() -> None:
    conflicting = {
        "version": 0,
        "childId": "c1",
        "childCreatedAt": 600,
        "mode": "one-shot",
        "label": "L",
    }
    with pytest.raises(
        SessionFormatUnsupportedMigrationError,
        match="conflicts with its parent catalog",
    ):
        _run_stage(
            _v3_migration([_v3_child_source()]),
            _header(3, "v3", isSeeded=True),
            4,
            _v3_catalog_rows(conflicting),
            kind="transformed",
        )


def test_migrating_restore_finish_reraises_current_validation() -> None:
    catalog = create_session_format_catalog_with_children([])
    decoder = _V01Codec(0).create_decoder(_header(0, "s"), "strict")
    collector = _Collector()
    migration = catalog._chain.create_stream(
        decoder.header, decoder.header_inherited_event_count, collector
    )

    def _boom(artifact: SessionFormatArtifact) -> SessionFormatArtifact:
        raise SessionFormatError("boom")

    restore = _MigratingSessionFormatRestore(
        decoder, 0, migration, collector, _boom, "current", 0, 4
    )
    restore.decode_row(
        {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}}
    )
    with pytest.raises(SessionFormatError, match="boom"):
        restore.finish()


def test_restore_v2_end_seed_without_inherited() -> None:
    artifact = _restore_released_v2_artifact(
        _artifact(
            2,
            [
                {"type": "turn/start", "seq": 0, "time": 100, "data": {"turn": 1}},
                {
                    "type": "session/end-seed",
                    "seq": 1,
                    "time": 101,
                    "data": {},
                },
            ],
        ),
        None,
    )
    assert [event["type"] for event in artifact.events] == [
        "turn/start",
        "session/end-seed",
    ]


def test_restore_v3_system_head_branches() -> None:
    events = _v3_valid_events()
    events.insert(3, _v3_system_event(3, 103, "append"))
    for index, event in enumerate(events):
        event["seq"] = index
        event["time"] = 100 + index
    assert (
        restore_released_v3_artifact(_artifact(3, events), RELEASED_V3_EVENT_TYPES)
        is not None
    )
    events = _v3_valid_events()
    events.insert(
        4, _v3_system_event(4, 104, {"op": "replace", "startSeq": 3, "endSeq": 3})
    )
    for index, event in enumerate(events):
        event["seq"] = index
        event["time"] = 100 + index
    assert (
        restore_released_v3_artifact(_artifact(3, events), RELEASED_V3_EVENT_TYPES)
        is not None
    )
    events = _v3_valid_events()
    events[3] = {
        **events[3],
        "surfaceOp": {"op": "replace", "startSeq": 0, "endSeq": 0},
    }
    assert (
        restore_released_v3_artifact(_artifact(3, events), RELEASED_V3_EVENT_TYPES)
        is not None
    )
    events = _v3_valid_events()
    events.append(
        {
            "type": "compaction/summary",
            "seq": 6,
            "time": 106,
            "data": {
                "compactionId": "c1",
                "summary": "s",
                "shadowedRange": {"start": 0, "end": 0},
                "shadowedSeqs": [0],
                "shadowedTokenCount": 1,
                "provider": "p",
                "model": "m",
            },
        }
    )
    assert (
        restore_released_v3_artifact(_artifact(3, events), RELEASED_V3_EVENT_TYPES)
        is not None
    )


def test_chain_name_duplicate_migration() -> None:
    same_name = SessionFormatMigration(
        name="session-format-v0-to-v1",
        from_version=1,
        to_version=2,
        migrate_header=SESSION_FORMAT_V1_TO_V2.migrate_header,
        create_stage=SESSION_FORMAT_V1_TO_V2.create_stage,
        validate_target_header=lambda header: header,
    )
    with pytest.raises(SessionFormatError, match="is duplicated"):
        SessionFormatChain(4, [SESSION_FORMAT_V0_TO_V1, same_name], assert_v4_header)


def test_migrate_v3_event_content_identity_branches() -> None:
    summary = migrate_v3_event_content(
        {
            "type": "compaction/summary",
            "seq": 0,
            "time": 100,
            "data": {
                "summary": [{"type": "widget"}],
                "rawOutput": [{"type": "text", "text": "x"}],
            },
        }
    )
    assert summary["data"]["summary"] == [{"type": "plugin:widget"}]
    assert summary["data"]["rawOutput"] == [{"type": "text", "text": "x"}]
    dispatch = migrate_v3_event_content(
        {
            "type": "tool/ptc-dispatch",
            "seq": 0,
            "time": 100,
            "data": {"content": [{"type": "text", "text": "x"}]},
        }
    )
    assert dispatch["data"]["content"] == [{"type": "text", "text": "x"}]
    queued = migrate_v3_event_content(
        {
            "type": "team/message/queued",
            "seq": 0,
            "time": 100,
            "data": {
                "message": {
                    "role": "user",
                    "id": "m",
                    "content": [{"type": "text", "text": "hi"}],
                    "source": {"kind": "user"},
                }
            },
        }
    )
    assert queued["data"]["message"]["content"] == [{"type": "text", "text": "hi"}]
    header = migrate_v3_event_content(
        {
            "type": "request/header",
            "seq": 0,
            "time": 100,
            "data": {
                "header": {"tools": [{"name": "x"}]},
                "reason": {"kind": "start"},
            },
        }
    )
    assert header["data"]["header"]["tools"] == [{"name": "x"}]


def test_lift_tool_result_non_user_message() -> None:
    row = _tool_result_row()
    assert lift_tool_result(row) is row


# ---------------------------------------------------------------------------
# P2-6 request reconstruction: load-layer admission
# ---------------------------------------------------------------------------


def _request_header_row(data: dict | None = None, **overrides: object) -> dict:
    row: dict = {
        "type": "request/header",
        "seq": 0,
        "time": 100,
        "data": data
        if data is not None
        else {
            "header": {"config": {"provider": "p", "model": "m"}},
            "reason": "initial",
        },
    }
    row.update(overrides)
    return row


def test_assert_v4_request_header_data_valid_rows_pass() -> None:
    """Verifies well-formed request/header rows pass load admission."""
    assert_v4_request_header_data(_request_header_row())
    for reason in ("initial", "resume", "change", "series"):
        assert_v4_request_header_data(
            _request_header_row(
                {
                    "header": {"config": {"provider": "p", "model": "m"}},
                    "reason": reason,
                }
            )
        )
    assert_v4_request_header_data(
        _request_header_row(
            {
                "header": {
                    "config": {
                        "provider": "p",
                        "model": "m",
                        "reasoningEffort": "high",
                        "maxTokens": 100,
                    },
                    "adapterDefaults": {
                        "reasoningEffort": True,
                        "maxTokens": True,
                    },
                    "tools": [
                        {
                            "type": "function",
                            "deferLoading": True,
                            "function": {"name": "a"},
                        }
                    ],
                },
                "reason": "initial",
                "startsSeries": True,
            }
        )
    )
    # The migration's husk header is grandfathered.
    assert_v4_request_header_data(
        _request_header_row({"header": {}, "reason": {"kind": "start"}})
    )
    # Rows of other types and non-object rows are not this function's business.
    assert_v4_request_header_data({"type": "turn/start", "seq": 0, "data": {}})
    assert_v4_request_header_data("not-a-row")
    assert_v4_request_header_data(
        {"type": "request/header", "seq": 0, "data": "not-an-object"}
    )
    assert_v4_request_header_data(
        _request_header_row({"header": "not-an-object", "reason": "initial"})
    )


def test_assert_v4_request_header_data_bad_rows_raise() -> None:
    """Verifies malformed request/header rows are refused at load."""
    bad: list[tuple[dict, str]] = [
        (
            _request_header_row({"header": {"config": {}}, "reason": "initial"}),
            "requires a config with a nonempty provider and model",
        ),
        (
            _request_header_row(
                {"header": {"config": {"provider": "p"}}, "reason": "initial"}
            ),
            "requires a config with a nonempty provider and model",
        ),
        (
            _request_header_row(
                {
                    "header": {
                        "config": {"provider": "p", "model": "m", "reasoningEffort": 3}
                    },
                    "reason": "initial",
                }
            ),
            "config reasoningEffort must be a nonempty string",
        ),
        (
            _request_header_row(
                {
                    "header": {"config": {"provider": "p", "model": "m"}},
                    "reason": "bogus",
                }
            ),
            "reason must be one of",
        ),
        (
            _request_header_row(
                {
                    "header": {"config": {"provider": "p", "model": "m"}},
                    "reason": {"kind": "initial"},
                }
            ),
            "reason must be one of",
        ),
        (
            _request_header_row(
                {
                    "header": {"config": {"provider": "p", "model": "m"}},
                    "reason": "initial",
                    "startsSeries": False,
                }
            ),
            "startsSeries must be true",
        ),
        (
            _request_header_row(
                {
                    "header": {
                        "config": {"provider": "p", "model": "m"},
                        "adapterDefaults": "yes",
                    },
                    "reason": "initial",
                }
            ),
            "adapterDefaults must be an object",
        ),
        (
            _request_header_row(
                {
                    "header": {
                        "config": {"provider": "p", "model": "m"},
                        "adapterDefaults": {"unknown": True},
                    },
                    "reason": "initial",
                }
            ),
            "carries unknown key",
        ),
        (
            _request_header_row(
                {
                    "header": {
                        "config": {"provider": "p", "model": "m"},
                        "adapterDefaults": {"reasoningEffort": False},
                    },
                    "reason": "initial",
                }
            ),
            "must be true when present",
        ),
        (
            _request_header_row(
                {
                    "header": {
                        "config": {"provider": "p", "model": "m"},
                        "adapterDefaults": {"maxTokens": True},
                    },
                    "reason": "initial",
                }
            ),
            "requires config maxTokens",
        ),
        (
            _request_header_row(
                {
                    "header": {"config": {"provider": "p", "model": "m"}, "tools": {}},
                    "reason": "initial",
                }
            ),
            "header tools must be an array",
        ),
        (
            _request_header_row(
                {
                    "header": {
                        "config": {"provider": "p", "model": "m"},
                        "tools": ["not-an-object"],
                    },
                    "reason": "initial",
                }
            ),
            "header tools must be objects",
        ),
        (
            _request_header_row(
                {
                    "header": {
                        "config": {"provider": "p", "model": "m"},
                        "tools": [{"type": "function", "deferLoading": False}],
                    },
                    "reason": "initial",
                }
            ),
            "header tool deferLoading must be true",
        ),
    ]
    for row, match in bad:
        with pytest.raises(SessionFormatError, match=match):
            assert_v4_request_header_data(row)


def _request_context_row(data: dict | None = None) -> dict:
    return {
        "type": "request/context",
        "seq": 0,
        "time": 100,
        "data": data if data is not None else {"provider": "p", "model": "m"},
    }


def test_assert_v4_request_context_data() -> None:
    """Verifies request/context rows pass or fail load admission."""
    assert_v4_request_context_data(_request_context_row())
    assert_v4_request_context_data(
        _request_context_row(
            {
                "provider": "p",
                "model": "m",
                "contextWindow": 4096,
                "systemPromptUpdate": "in-history",
            }
        )
    )
    assert_v4_request_context_data({"type": "turn/start", "seq": 0, "data": {}})
    assert_v4_request_context_data(
        {"type": "request/context", "seq": 0, "data": "not-an-object"}
    )
    bad: list[tuple[dict, str]] = [
        (_request_context_row({"provider": "", "model": "m"}), "nonempty provider"),
        (_request_context_row({"provider": "p", "model": ""}), "nonempty provider"),
        (
            _request_context_row({"provider": "p", "model": "m", "contextWindow": 0}),
            "contextWindow must be a positive safe integer",
        ),
        (
            _request_context_row({"provider": "p", "model": "m", "contextWindow": 1.5}),
            "contextWindow must be a positive safe integer",
        ),
        (
            _request_context_row(
                {"provider": "p", "model": "m", "systemPromptUpdate": "replace"}
            ),
            'systemPromptUpdate must be "in-history"',
        ),
    ]
    for row, match in bad:
        with pytest.raises(SessionFormatError, match=match):
            assert_v4_request_context_data(row)


def _settlement_row(type: str = "assistant/message", data: dict | None = None) -> dict:
    return {
        "type": type,
        "seq": 0,
        "time": 100,
        "data": data
        if data is not None
        else {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "a",
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
                "source": {"kind": "model", "provider": "p", "model": "m"},
            },
            "stream": [],
        },
    }


def test_assert_v4_assistant_settlement_data() -> None:
    """Verifies assistant settlement rows pass or fail load admission."""
    assert_v4_assistant_settlement_data(_settlement_row())
    assert_v4_assistant_settlement_data(
        _settlement_row("assistant/attempt", {"turn": 1, "step": 1, "stream": []})
    )
    # A source without provider/model is grandfathered at load.
    assert_v4_assistant_settlement_data(
        _settlement_row(
            data={
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "a",
                    "role": "assistant",
                    "content": [],
                    "source": {"kind": "model"},
                },
                "stream": [],
            }
        )
    )
    assert_v4_assistant_settlement_data({"type": "turn/start", "seq": 0, "data": {}})
    assert_v4_assistant_settlement_data(
        {"type": "assistant/message", "seq": 0, "data": "not-an-object"}
    )
    message = _settlement_row()["data"]["message"]
    bad: list[tuple[dict, str]] = [
        (
            _settlement_row(
                data={
                    "turn": 1,
                    "step": 1,
                    "message": "not-an-object",
                    "stream": [],
                }
            ),
            "requires a message object",
        ),
        (
            _settlement_row(
                data={"turn": -1, "step": 1, "message": message, "stream": []}
            ),
            "turn must be a non-negative safe integer",
        ),
        (
            _settlement_row(
                data={"turn": 1, "step": 1.5, "message": message, "stream": []}
            ),
            "step must be a non-negative safe integer",
        ),
        (
            _settlement_row(
                data={"turn": 1, "step": 1, "message": message, "stream": {}}
            ),
            "stream must be an array",
        ),
        (
            _settlement_row(
                data={
                    "turn": 1,
                    "step": 1,
                    "message": message,
                    "stream": [],
                    "tool_calls": [],
                }
            ),
            "must not carry tool_calls",
        ),
        (
            _settlement_row(
                data={
                    "turn": 1,
                    "step": 1,
                    "message": message,
                    "stream": [],
                    "interrupted": False,
                }
            ),
            "interrupted must be true",
        ),
        (
            _settlement_row(
                data={
                    "turn": 1,
                    "step": 1,
                    "message": {**message, "id": ""},
                    "stream": [],
                }
            ),
            "requires an identified model message",
        ),
        (
            _settlement_row(
                data={
                    "turn": 1,
                    "step": 1,
                    "message": {**message, "role": "user"},
                    "stream": [],
                }
            ),
            "requires an identified model message",
        ),
        (
            _settlement_row(
                data={
                    "turn": 1,
                    "step": 1,
                    "message": {**message, "content": "hi"},
                    "stream": [],
                }
            ),
            "requires an identified model message",
        ),
        (
            _settlement_row(
                data={
                    "turn": 1,
                    "step": 1,
                    "message": {**message, "source": {"kind": "user"}},
                    "stream": [],
                }
            ),
            "requires an identified model message",
        ),
        (
            _settlement_row(
                data={
                    "turn": 1,
                    "step": 1,
                    "message": {
                        **message,
                        "source": {"kind": "model", "provider": "p"},
                    },
                    "stream": [],
                }
            ),
            "requires a nonempty provider and model",
        ),
    ]
    for row, match in bad:
        with pytest.raises(SessionFormatError, match=match):
            assert_v4_assistant_settlement_data(row)
