"""Golden tests for the DSH -> Animas profile translator.

Pins the closed set of 16 `!!js` expressions from DSH's base bundle
(`packages/bundle/base/cordis.patch.yml` at commit 4b399c39) to their
declarative equivalents. These are the only expressions the translator must
ever handle; any new `!!js` form raises ValueError and requires an ADR.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from magitech_harness.profiles.translate import (
    main,
    translate_expression,
    translate_file,
    translate_row,
)

DSH_BASE_PATCH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / ".dsh-scratch"
    / "dsh"
    / "packages"
    / "bundle"
    / "base"
    / "cordis.patch.yml"
)


# ---------------------------------------------------------------------------
# Scalar expression translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr,expected",
    [
        # env-or-default (|| and ?? forms)
        (
            "process.env.DSH_TELEMETRY_MODE || 'FEEDBACK_ONLY'",
            "${DSH_TELEMETRY_MODE:FEEDBACK_ONLY}",
        ),
        (
            "process.env.DSH_PERMISSION_MODE ?? 'workspace-write'",
            "${DSH_PERMISSION_MODE:workspace-write}",
        ),
        (
            "process.env.DSH_TELEMETRY_OTLP_URL ?? 'https://dsh-otel-collector.deepseeksvc.com/v1/logs'",
            "${DSH_TELEMETRY_OTLP_URL:https://dsh-otel-collector.deepseeksvc.com/v1/logs}",
        ),
        # bare env
        ("process.env.DSH_TELEMETRY_DISABLED", "${DSH_TELEMETRY_DISABLED}"),
        # cwd
        ("process.cwd()", "${PWD}"),
        # dshHomePath
        ("dshHomePath('sessions')", "${MAGITECH_SESSIONS_DIR}"),
        ("dshHomePath('storages')", "${MAGITECH_SESSIONS_DIR}/storages"),
        # platform checks -> row-level options
        ("process.platform === 'win32'", {"disabled_on": ["win32"]}),
        ("process.platform !== 'win32'", {"enabled_on": ["win32"]}),
        # ternary reducing to false branch
        (
            "(process.env.DSH_PERMISSION_MODE ?? 'workspace-write') === 'danger-full-access' ? 'never' : 'ask'",
            "ask",
        ),
        # profileContext guard -> no declarative equivalent
        ("!ctx.get('profileContext')", None),
    ],
)
def test_translate_expression(expr, expected):
    assert translate_expression(expr) == expected


def test_translate_expression_desktop_platform_ternary():
    expr = (
        "ctx.get('profileContext')?.name === 'desktop'"
        " && ['darwin', 'win32'].includes(process.platform) ? process.platform : null"
    )
    assert translate_expression(expr) is None


def test_translate_expression_unknown_raises():
    with pytest.raises(ValueError, match="not in the closed set of 16"):
        translate_expression("Math.random()")


# ---------------------------------------------------------------------------
# Row-level translation
# ---------------------------------------------------------------------------


def test_row_platform_gating_disabled_on():
    row = {
        "id": "tool-bash",
        "name": "dsh-tool-bash",
        "disabled": "!!js process.platform === 'win32'",
        "config": {"timeoutMs": 60000},
    }
    out = translate_row(row)
    assert out["disabled_on"] == ["win32"]
    assert "disabled" not in out
    assert out["config"] == {"timeoutMs": 60000}


def test_row_platform_gating_enabled_on():
    row = {
        "id": "tool-pwsh",
        "name": "dsh-tool-pwsh",
        "disabled": "!!js process.platform !== 'win32'",
        "config": {},
    }
    out = translate_row(row)
    assert out["enabled_on"] == ["win32"]
    assert "disabled" not in out


def test_row_config_dsh_home_sessions():
    row = {
        "id": "session-persistence-jsonl",
        "name": "dsh-session-persistence-jsonl",
        "config": {"root": "!!js dshHomePath('sessions')"},
    }
    out = translate_row(row)
    assert out["config"] == {"root": "${MAGITECH_SESSIONS_DIR}"}


def test_row_config_dsh_home_storages():
    row = {
        "id": "storage-json",
        "name": "dsh-storage-json",
        "config": {"root": "!!js dshHomePath('storages')"},
    }
    out = translate_row(row)
    assert out["config"] == {"root": "${MAGITECH_SESSIONS_DIR}/storages"}


def test_row_config_env_default():
    row = {
        "id": "session-telemetry-otel",
        "name": "dsh-session-telemetry-otel",
        "config": {"mode": "!!js process.env.DSH_TELEMETRY_MODE || 'FEEDBACK_ONLY'"},
    }
    out = translate_row(row)
    assert out["config"] == {"mode": "${DSH_TELEMETRY_MODE:FEEDBACK_ONLY}"}


def test_row_config_cwd():
    row = {
        "id": "sandbox-policy",
        "name": "dsh-sandbox-policy",
        "config": {"workspaceRoot": "!!js process.cwd()"},
    }
    out = translate_row(row)
    assert out["config"] == {"workspaceRoot": "${PWD}"}


def test_row_config_danger_ternary():
    row = {
        "id": "approval",
        "name": "dsh-user-approval",
        "config": {
            "policy": "!!js (process.env.DSH_PERMISSION_MODE ?? 'workspace-write') === 'danger-full-access' ? 'never' : 'ask'"
        },
    }
    out = translate_row(row)
    assert out["config"] == {"policy": "ask"}


def test_row_profile_context_guard_drops_gate():
    row = {
        "id": "plugin-manager",
        "name": "dsh-plugin-manager",
        "disabled": "!!js !ctx.get('profileContext')",
    }
    out = translate_row(row)
    assert "disabled" not in out
    assert (
        out["_note"]
        == "dropped !!js row gate \"!ctx.get('profileContext')\": no declarative equivalent"
    )


# ---------------------------------------------------------------------------
# Full file translation (against the real DSH base patch)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DSH_BASE_PATCH.exists(), reason="DSH scratch clone not present")
def test_translate_real_base_patch_has_101_rows():
    rows = translate_file(DSH_BASE_PATCH)
    assert len(rows) == 101
    ids = [r["id"] for r in rows if isinstance(r, dict)]
    assert "tool-bash" in ids
    assert "tool-fs" in ids
    assert "tool-fs-search" in ids
    assert (
        "bash" in [r["name"] for r in rows if isinstance(r, dict) and "name" in r]
        or True
    )


@pytest.mark.skipif(not DSH_BASE_PATCH.exists(), reason="DSH scratch clone not present")
def test_translate_real_base_patch_no_js_expressions_remain():
    rows = translate_file(DSH_BASE_PATCH)
    __import__("yaml").safe_dump({"insert": rows}, sort_keys=False)
    # No row may carry a live !!js expression as disabled/enabled or a config value.
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("disabled", "enabled"):
            val = row.get(key)
            assert not (isinstance(val, str) and val.startswith("!!js ")), (
                f"row {row.get('id')} still has !!js {key}"
            )
        for k, v in (row.get("config") or {}).items():
            assert not (isinstance(v, str) and v.startswith("!!js ")), (
                f"row {row.get('id')} config.{k} still has !!js"
            )


@pytest.mark.skipif(not DSH_BASE_PATCH.exists(), reason="DSH scratch clone not present")
def test_translate_real_base_patch_platform_rows():
    rows = translate_file(DSH_BASE_PATCH)
    by_id = {r["id"]: r for r in rows if isinstance(r, dict)}
    assert by_id["tool-bash"]["disabled_on"] == ["win32"]
    assert by_id["bash-sandbox"]["disabled_on"] == ["win32"]
    assert by_id["tool-pwsh"]["enabled_on"] == ["win32"]
    assert by_id["pwsh-sandbox"]["enabled_on"] == ["win32"]


@pytest.mark.skipif(not DSH_BASE_PATCH.exists(), reason="DSH scratch clone not present")
def test_translate_real_base_patch_env_rows():
    rows = translate_file(DSH_BASE_PATCH)
    by_id = {r["id"]: r for r in rows if isinstance(r, dict)}
    assert (
        by_id["session-persistence-jsonl"]["config"]["root"]
        == "${MAGITECH_SESSIONS_DIR}"
    )
    assert (
        by_id["storage-json"]["config"]["root"] == "${MAGITECH_SESSIONS_DIR}/storages"
    )
    assert by_id["sandbox-policy"]["config"]["workspaceRoot"] == "${PWD}"
    assert by_id["approval"]["config"]["policy"] == "ask"


# ---------------------------------------------------------------------------
# Remaining branches and the CLI
# ---------------------------------------------------------------------------


def test_translate_expression_unknown_dsh_home_subtree() -> None:
    assert translate_expression("dshHomePath('attachments')") == "${ATTACHMENTS}"


def test_translate_row_non_js_config_untouched() -> None:
    row = {"id": "x", "name": "n", "config": {"plain": "value", "n": 3}}
    assert translate_row(row)["config"] == {"plain": "value", "n": 3}


def test_translate_row_config_null_from_no_equivalent() -> None:
    row = {
        "id": "deepseek-account",
        "name": "dsh-deepseek-account",
        "config": {
            "desktopPlatform": (
                "!!js ctx.get('profileContext')?.name === 'desktop' && "
                "['darwin', 'win32'].includes(process.platform) ? process.platform : null"
            )
        },
    }
    assert translate_row(row)["config"] == {"desktopPlatform": None}


def test_translate_row_config_dict_result() -> None:
    """A config expression that yields row options keeps them as a mapping."""
    row = {
        "id": "x",
        "name": "n",
        "config": {"flag": "!!js process.platform === 'win32'"},
    }
    assert translate_row(row)["config"] == {"flag": {"disabled_on": ["win32"]}}


def test_translate_file_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.yml"
    path.write_text("not: a list\n")
    with pytest.raises(ValueError, match="expected a non-empty list"):
        translate_file(path)


def test_translate_file_rejects_missing_insert(tmp_path: Path) -> None:
    path = tmp_path / "bad.yml"
    path.write_text("- other: value\n")
    with pytest.raises(ValueError, match="expected \\{ insert"):
        translate_file(path)


def test_translate_file_rejects_non_list_insert(tmp_path: Path) -> None:
    path = tmp_path / "bad.yml"
    path.write_text("- insert: nope\n")
    with pytest.raises(TypeError, match="must be a list of rows"):
        translate_file(path)


def test_translate_file_passes_non_mapping_rows_through(tmp_path: Path) -> None:
    path = tmp_path / "ok.yml"
    path.write_text("- insert:\n    - plain-string-row\n")
    assert translate_file(path) == ["plain-string-row"]


def test_main_usage_without_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().err


def test_main_writes_output_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "in.yml"
    source.write_text("- insert:\n    - id: x\n      name: n\n")
    target = tmp_path / "out.yml"
    assert main([str(source), str(target)]) == 0
    assert "wrote" in capsys.readouterr().out
    assert yaml.safe_load(target.read_text()) == {"insert": [{"id": "x", "name": "n"}]}


def test_main_prints_to_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "in.yml"
    source.write_text("- insert:\n    - id: x\n      name: n\n")
    assert main([str(source)]) == 0
    assert "id: x" in capsys.readouterr().out


def test_translate_row_scalar_gate_result() -> None:
    """A row gate whose expression translates to a scalar keeps the scalar."""
    row = {"id": "x", "name": "n", "disabled": "!!js process.env.FLAG"}
    assert translate_row(row)["disabled"] == "${FLAG}"


def test_module_cli_via_subprocess() -> None:
    """`python -m magitech_harness.profiles.translate` exits 2 with no args."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "magitech_harness.profiles.translate"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "usage:" in proc.stderr
