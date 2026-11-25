"""Translate DSH `cordis.patch.yml` `!!js` expressions to declarative Animas-native profile rows.

This is a *port-time* translator: it reads a DSH base bundle patch that contains
`!!js` config expressions and emits the equivalent declarative profile YAML that
Animas can load without any runtime expression evaluator (ADR 0022 — Animas stays
evaluator-free).

The 16 `!!js` expressions in DSH's base bundle are a closed set. Each maps to a
declarative equivalent:

  * `!!js "!ctx.get('profileContext')"`   -> row is profile-gated; for the Python
      port this means the row lives in mth-components or is out of scope, so the
      translator leaves it as a comment marker rather than emitting a row.
  * `!!js dshHomePath('sessions')`         -> ${MAGITECH_SESSIONS_DIR}
  * `!!js dshHomePath('storages')`         -> ${MAGITECH_SESSIONS_DIR}/storages
  * `!!js process.env.X || 'default'`      -> ${X:default}
  * `!!js process.env.X ?? 'default'`      -> ${X:default}
  * `!!js process.cwd()`                   -> ${PWD}
  * `!!js process.platform === 'win32'`    -> disabled_on: [win32]
  * `!!js process.platform !== 'win32'`    -> enabled_on: [win32]
  * `!!js "<ternary>"`                     -> static default branch value
  * `!!js "ctx.get('profileContext')?.name === 'desktop' && ... ? process.platform : null"` -> null value

Never adds a runtime expression evaluator to Animas. If literal DSH YAML
compatibility is ever required, stop and ask for a new ADR superseding 0022.

Usage:
    python3 -m magitech_harness.profiles.translate <input.yml> [output.yml]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

__all__ = ["main", "translate_expression", "translate_file"]

# ---------------------------------------------------------------------------
# Expression -> declarative equivalent
# ---------------------------------------------------------------------------

# Env-or-default: process.env.VAR || 'default'  and  process.env.VAR ?? 'default'
_ENV_OR_DEFAULT = re.compile(
    r"^process\.env\.([A-Za-z_][A-Za-z0-9_]*)\s*\|\|\s*'([^']*)'$"
    r"|^process\.env\.([A-Za-z_][A-Za-z0-9_]*)\s*\?\?\s*'([^']*)'$"
)

# Bare env: process.env.VAR
_BARE_ENV = re.compile(r"^process\.env\.([A-Za-z_][A-Za-z0-9_]*)$")

# Platform checks
_PLATFORM_WIN32_EQ = re.compile(r"^process\.platform\s*===\s*'win32'$")
_PLATFORM_WIN32_NE = re.compile(r"^process\.platform\s*!==\s*'win32'$")

# process.cwd()
_CWD = re.compile(r"^process\.cwd\(\)$")

# dshHomePath(...)
_DSH_HOME = re.compile(r"^dshHomePath\('([^']*)'\)$")

# Ternary that reduces to the false branch (danger-full-access check).
# `(env ?? 'default') === 'danger-full-access' ? 'never' : 'ask'` -> 'ask'
_DANGER_TERNARY = re.compile(
    r"^\(process\.env\.([A-Za-z_][A-Za-z0-9_]*)\s*\?\?\s*'([^']*)'\)\s*===\s*'danger-full-access'\s*\?\s*'([^']*)'\s*:\s*'([^']*)'$"
)

# ctx.get('profileContext') guard - not expressible declaratively; leave as marker.
_PROFILE_CTX_GUARD = re.compile(r"^!ctx\.get\('profileContext'\)$")

# deepseek-account desktopPlatform ternary:
# ctx.get('profileContext')?.name === 'desktop' && ['darwin','win32'].includes(process.platform) ? process.platform : null
# Evaluates to null in every non-desktop case; null is the default for an unset value,
# so the declarative equivalent is simply `desktopPlatform: null` (or omit the key).
_DESKTOP_PLATFORM_TERNARY = re.compile(
    r"^ctx\.get\('profileContext'\)\?\.name\s*===\s*'desktop'\s*&&\s*"
    r"\['darwin',\s*'win32'\]\.includes\(process\.platform\)\s*\?\s*process\.platform\s*:\s*null$"
)


def translate_expression(expr: str) -> dict[str, Any] | str | None:
    """Translate one `!!js` expression string to a declarative value or row options.

    Returns:
        - A string replacement (for scalar config values).
        - A dict of row-level options (e.g. ``{"disabled_on": ["win32"]}``) when
          the expression gates the whole row.
        - ``None`` when the expression has no declarative equivalent (e.g.
          ``!ctx.get('profileContext')``); the caller must decide whether to drop
          or comment the row.

    Raises:
        ValueError: if the expression is not in the closed set of 16 known forms.
    """
    expr = expr.strip()

    m = _ENV_OR_DEFAULT.match(expr)
    if m:
        var = m.group(1) or m.group(3)
        default = m.group(2) or m.group(4)
        return f"${{{var}:{default}}}"

    m = _BARE_ENV.match(expr)
    if m:
        return f"${{{m.group(1)}}}"

    m = _PLATFORM_WIN32_EQ.match(expr)
    if m:
        return {"disabled_on": ["win32"]}

    m = _PLATFORM_WIN32_NE.match(expr)
    if m:
        return {"enabled_on": ["win32"]}

    m = _CWD.match(expr)
    if m:
        return "${PWD}"

    m = _DSH_HOME.match(expr)
    if m:
        sub = m.group(1)
        if sub == "sessions":
            return "${MAGITECH_SESSIONS_DIR}"
        if sub == "storages":
            return "${MAGITECH_SESSIONS_DIR}/storages"
        return f"${{{sub.upper()}}}"

    m = _DANGER_TERNARY.match(expr)
    if m:
        # Default branch: ask (the non-danger-full-access case).
        return m.group(4)

    m = _PROFILE_CTX_GUARD.match(expr)
    if m:
        return None

    m = _DESKTOP_PLATFORM_TERNARY.match(expr)
    if m:
        # desktopPlatform defaults to null; no declarative config value needed.
        return None

    raise ValueError(f"unknown !!js expression (not in the closed set of 16): {expr!r}")


# ---------------------------------------------------------------------------
# Row translation
# ---------------------------------------------------------------------------

KNOWN_JS_ROWS = frozenset(
    {
        # Rows whose !!js expression gates the whole row (disabled/enabled).
        "plugin-manager",
        "config-editor",
        "settings",
        "tool-bash",
        "bash-sandbox",
        "tool-pwsh",
        "pwsh-sandbox",
        # Rows whose !!js expression is a config value.
        "session-persistence-jsonl",
        "storage-json",
        "session-telemetry-otel",
        "sandbox-policy",
        "approval",
        "deepseek-account",
    }
)


def translate_row(row: dict[str, Any]) -> dict[str, Any]:
    """Translate one profile row, converting any `!!js` values/options.

    Mutates a copy; returns a new dict.
    """
    out: dict[str, Any] = dict(row)

    # Top-level `disabled` / `enabled` can be a `!!js` expression.
    for key in ("disabled", "enabled"):
        if key in out and isinstance(out[key], str) and out[key].startswith("!!js "):
            expr = out[key][len("!!js ") :]
            result = translate_expression(expr)
            if isinstance(result, dict):
                # Row-level options (disabled_on / enabled_on).
                out.pop(key, None)
                out.update(result)
            elif result is None:
                # No declarative equivalent: drop the gating key and leave a note.
                out.pop(key, None)
                out["_note"] = (
                    f"dropped !!js row gate {expr!r}: no declarative equivalent"
                )
            else:
                out[key] = result

    # Config values can contain `!!js` expressions (as scalars or nested).
    if "config" in out and isinstance(out["config"], dict):
        out["config"] = _translate_config(out["config"])

    return out


def _translate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, str) and value.startswith("!!js "):
            result = translate_expression(value[len("!!js ") :])
            if result is None:
                # Null value (e.g. desktopPlatform default). Keep the key as null.
                out[key] = None
            elif isinstance(result, dict):
                out[key] = result
            else:
                out[key] = result
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# File translation
# ---------------------------------------------------------------------------

_JSSCALAR = re.compile(r"^!!js (.*)$")


def _js_constructor(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> str:
    """Capture `!!js <expr>` as the raw expression string (with the tag prefix)."""
    return f"!!js {loader.construct_scalar(node)}"


def load_dsh_yaml(path: Path | str) -> Any:
    """Load a DSH cordis.patch.yml, capturing `!!js` expressions as strings."""
    yaml.SafeLoader.add_constructor("tag:yaml.org,2002:js", _js_constructor)
    try:
        return yaml.safe_load(Path(path).read_text())
    finally:
        yaml.SafeLoader.yaml_constructors.pop("tag:yaml.org,2002:js", None)


def translate_file(path: Path | str) -> list[dict[str, Any]]:
    """Translate a DSH cordis.patch.yml file into declarative profile rows.

    The top-level document is expected to be a list with one item containing an
    ``insert`` list of rows (the DSH bundle patch shape).
    """
    data = load_dsh_yaml(path)
    if not isinstance(data, list) or not data:
        raise ValueError(f"expected a non-empty list at the top of {path}")
    bundle = data[0]
    if not isinstance(bundle, dict) or "insert" not in bundle:
        raise ValueError(f"expected {{ insert: [...] }} at the top of {path}")
    rows = bundle["insert"]
    if not isinstance(rows, list):
        raise TypeError("'insert' must be a list of rows")
    return [translate_row(dict(r)) if isinstance(r, dict) else r for r in rows]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "usage: python -m magitech_harness.profiles.translate <input.yml> [output.yml]",
            file=sys.stderr,
        )
        return 2
    input_path = Path(argv[0])
    output_path = Path(argv[1]) if len(argv) > 1 else None

    rows = translate_file(input_path)
    doc = {"insert": rows}
    text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)

    if output_path is not None:
        output_path.write_text(text)
        print(f"wrote {output_path} ({len(rows)} rows)")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
