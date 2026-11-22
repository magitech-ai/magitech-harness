"""Unit tests for the unified author schema DSL and define_tool helper."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import patch

import pytest

from magitech_harness.tools.json_schema import (
    JsonSchemaError,
    validate_json_schema_value,
)
from magitech_harness.tools.schema import (
    ToolArgsError,
    define_tool,
    parameter_schema_spec_to_json_schema,
    project_tool_schema,
    validate_args,
    validate_value,
    value_schema_spec_to_json_schema,
)


def test_compiles_value_roots_and_json_node() -> None:
    """Compiles every ValueSchemaSpec kind including author-only 'json'."""
    assert value_schema_spec_to_json_schema(
        {"type": "string", "enum": ["a", "b"], "const": "a"}
    ) == {"type": "string", "enum": ["a", "b"], "const": "a"}
    assert value_schema_spec_to_json_schema({"type": "number"}) == {"type": "number"}
    assert value_schema_spec_to_json_schema({"type": "integer"}) == {"type": "integer"}
    assert value_schema_spec_to_json_schema({"type": "boolean"}) == {"type": "boolean"}
    assert value_schema_spec_to_json_schema({"type": "null"}) == {"type": "null"}
    assert value_schema_spec_to_json_schema(
        {"type": "array", "items": {"type": "json"}}
    ) == {"type": "array", "items": {}}
    assert value_schema_spec_to_json_schema(
        {"type": "object", "additionalProperties": False, "properties": {}}
    ) == {"type": "object", "additionalProperties": False, "properties": {}}
    assert value_schema_spec_to_json_schema(
        {
            "type": "json",
            "description": "anything",
            "title": "Any JSON",
            "default": None,
            "examples": [{"nested": True}],
        }
    ) == {
        "description": "anything",
        "title": "Any JSON",
        "default": None,
        "examples": [{"nested": True}],
    }
    assert value_schema_spec_to_json_schema(
        {"oneOf": [{"type": "string"}, {"type": "null"}]}
    ) == {"oneOf": [{"type": "string"}, {"type": "null"}]}


def test_compiles_parameter_schema_with_open_root_and_explicit_nested() -> None:
    """Compiles parameter maps, extracting required while preserving nested openness."""
    assert parameter_schema_spec_to_json_schema({}) == {
        "type": "object",
        "properties": {},
    }
    compiled = parameter_schema_spec_to_json_schema(
        {
            "closed": {
                "type": "object",
                "additionalProperties": False,
                "required": True,
                "properties": {"id": {"type": "integer", "required": True}},
            },
            "open": {"type": "object", "additionalProperties": True},
        }
    )
    assert compiled == {
        "type": "object",
        "properties": {
            "closed": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            },
            "open": {"type": "object", "additionalProperties": True},
        },
        "required": ["closed"],
    }


def test_rejects_invalid_and_cyclic_author_schemas() -> None:
    """Rejects forged/unsupported author schema forms via JsonSchemaError."""
    for bad in (
        {"type": "object"},
        {"oneOf": [{"type": "string"}]},
        {"type": "number", "enum": ["1"]},
        {"type": "string", "enum": ["a"], "const": "b"},
        {"type": "integer", "const": 1.5},
        {"type": "json", "default": float("nan")},
        {"type": "array", "items": {"type": "string", "required": True}},
        {"type": "array", "items": 42},
        {"type": "string", "extra": True},
        {"oneOf": "not-an-array"},
        {"type": "string", "enum": "a"},
        {},
        None,
    ):
        with pytest.raises(JsonSchemaError):
            value_schema_spec_to_json_schema(bad)

    with pytest.raises(JsonSchemaError, match="cannot declare both type and oneOf"):
        value_schema_spec_to_json_schema(
            {"type": "string", "oneOf": [{"type": "string"}, {"type": "null"}]}
        )

    for bad_params in (
        {"value": {"type": "string", "required": False}},
        None,
        {"bad": 42},
    ):
        with pytest.raises(JsonSchemaError):
            parameter_schema_spec_to_json_schema(bad_params)

    cyclic_arr: dict[str, object] = {"type": "array"}
    cyclic_arr["items"] = cyclic_arr
    with pytest.raises(JsonSchemaError, match="circular"):
        value_schema_spec_to_json_schema(cyclic_arr)

    props: dict[str, object] = {}
    props["self"] = {
        "type": "object",
        "additionalProperties": True,
        "properties": props,
    }
    with pytest.raises(JsonSchemaError, match="circular"):
        parameter_schema_spec_to_json_schema(props)


def test_compiles_deeply_nested_author_unions_iteratively() -> None:
    """5000-deep author union compiles iteratively without hitting call stack limit."""
    spec: dict[str, object] = {"type": "string"}
    for _ in range(5000):
        spec = {"oneOf": [spec, {"type": "null"}]}
    compiled = value_schema_spec_to_json_schema(spec)
    cursor: dict[str, object] = compiled
    layers = 0
    while "oneOf" in cursor:
        cursor = cursor["oneOf"][0]  # type: ignore[index]
        layers += 1
    assert layers == 5000
    assert cursor == {"type": "string"}


def test_validate_args_paths_and_edge_cases() -> None:
    """validate_args validates types, required fields, arrays, objects, no defaults."""
    spec = {
        "a": {"type": "string", "required": True, "default": "fallback"},
        "b": {"type": "integer"},
        "arr": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"id": {"type": "integer", "required": True}},
            },
        },
    }
    assert validate_args(spec, {"a": "ok", "extra": True}) == []
    assert validate_args(spec, {}) == ['missing required property "a"']
    assert validate_args(spec, {"a": "ok", "arr": [{"id": "bad"}]}) == [
        '"arr[0].id" must be an integer'
    ]
    assert validate_args(spec, "not-an-object") == ['"arguments" must be an object']


def test_define_tool_validates_execution_presentation_timeout_and_projection() -> None:
    """define_tool validates execute strictly, soft-validates display/concurrency, validates timeoutMs."""
    err = ToolArgsError(['missing required property "a"', '"b" must be a number'])
    assert err.code == "INVALID_ARGS"
    assert err.violations == ['missing required property "a"', '"b" must be a number']
    assert (
        err.message
        == 'invalid arguments: missing required property "a"; "b" must be a number'
    )

    counts = {"conc": 0}

    def _conc(args: Mapping[str, Any]) -> bool:
        counts["conc"] += 1
        return bool(args.get("uppercase"))

    tool = define_tool(
        {
            "name": "typed-echo",
            "description": "Typed echo tool",
            "parameters": {
                "text": {"type": "string", "required": True},
                "uppercase": {"type": "boolean"},
            },
            "output": {
                "schema": {"type": "string"},
                "render": lambda _args, value: [{"type": "text", "text": value}],
            },
            "isConcurrencySafe": _conc,
            "presentCall": lambda args: (
                {"kind": "other", "label": args["text"]}
                if isinstance(args.get("text"), str)
                else None
            ),
            "presentResult": lambda args, _res: {
                "kind": "generic",
                "text": args["text"],
            },
            "timeoutMs": 500,
            "deferLoading": True,
            "execute": lambda args, _ctx: (
                args["text"].upper() if args.get("uppercase") else args["text"]
            ),
        }
    )

    assert tool["timeoutMs"] == 500
    assert tool["isConcurrencySafe"]({"uppercase": True}) is False
    assert counts["conc"] == 0
    assert tool["presentCall"]({}) is None
    assert (
        tool["presentResult"]({"wrong": 1}, {"content": [], "isError": False}) is None
    )
    assert tool["isConcurrencySafe"]({"text": "hi", "uppercase": True}) is True
    assert counts["conc"] == 1
    assert tool["presentCall"]({"text": "hi"}) == {"kind": "other", "label": "hi"}

    assert tool["execute"]({"text": "hi", "uppercase": True}, {}) == "HI"
    with pytest.raises(ToolArgsError) as exc_info:
        tool["execute"]({}, {})
    assert exc_info.value.code == "INVALID_ARGS"

    with pytest.raises(ValueError, match="timeoutMs"):
        define_tool(
            {
                "name": "bad-timeout",
                "description": "d",
                "parameters": {},
                "output": {"schema": {"type": "null"}, "render": lambda _a, _v: []},
                "timeoutMs": 0,
                "execute": lambda _a, _c: None,
            }
        )
    with pytest.raises(ValueError, match="timeoutMs"):
        define_tool(
            {
                "name": "nan-timeout",
                "description": "d",
                "parameters": {},
                "output": {"schema": {"type": "null"}, "render": lambda _a, _v: []},
                "timeoutMs": float("inf"),
                "execute": lambda _a, _c: None,
            }
        )
    with pytest.raises(ValueError, match="timeoutMs"):
        define_tool(
            {
                "name": "bool-timeout",
                "description": "d",
                "parameters": {},
                "output": {"schema": {"type": "null"}, "render": lambda _a, _v: []},
                "timeoutMs": True,
                "execute": lambda _a, _c: None,
            }
        )

    full = project_tool_schema(tool, include_defer_loading=True)
    assert full == {
        "name": "typed-echo",
        "description": "Typed echo tool",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "uppercase": {"type": "boolean"},
            },
            "required": ["text"],
        },
        "deferLoading": True,
    }
    stripped = project_tool_schema(tool, include_defer_loading=False)
    assert "deferLoading" not in stripped
    assert set(stripped.keys()) == {"name", "description", "parameters"}
    shallow = project_tool_schema(tool, detach_parameters=False)
    assert shallow["parameters"] is tool["parameters"]

    # Non-lossless parameters fail loudly; 5000-deep schema projects without RecursionError
    with pytest.raises(ValueError, match="lossless JSON"):
        project_tool_schema(
            {
                "name": "bad_params",
                "description": "d",
                "parameters": {"type": "object", "default": float("nan")},
            }
        )
    deep_param: dict[str, Any] = {"type": "string"}
    for _ in range(5000):
        deep_param = {"oneOf": [deep_param, {"type": "null"}]}
    projected_deep = project_tool_schema(
        {"name": "deep", "description": "d", "parameters": deep_param}
    )
    assert projected_deep["parameters"] is not deep_param


def test_schema_dsl_additional_branches_and_kwarg_define_tool() -> None:
    """Covers nested required validation, parameter oneOf/cycles, validate_value, and errors."""
    with pytest.raises(JsonSchemaError, match="required"):
        value_schema_spec_to_json_schema(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"bad": {"type": "string", "required": False}},
            }
        )
    with pytest.raises(JsonSchemaError, match="properties"):
        value_schema_spec_to_json_schema(
            {"type": "object", "additionalProperties": False, "properties": "not-map"}
        )

    assert value_schema_spec_to_json_schema(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "opt": {"type": "string"},
                "arr": {"type": "array"},
            },
        }
    ) == {
        "type": "object",
        "additionalProperties": False,
        "properties": {"opt": {"type": "string"}, "arr": {"type": "array"}},
    }

    assert validate_value({"type": "string"}, "ok") == []
    assert validate_value({"type": "string"}, 42, path="field") == [
        '"field" must be a string'
    ]

    param_schema = parameter_schema_spec_to_json_schema(
        {
            "choice": {"oneOf": [{"type": "string"}, {"type": "number"}]},
            "nested": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"k": {"type": "string"}},
            },
        }
    )
    assert validate_json_schema_value(param_schema, {"choice": "a"}) == []

    bad_nested_props: dict[str, Any] = {
        "nested": {
            "type": "object",
            "additionalProperties": False,
            "properties": "bad",
        }
    }
    with pytest.raises(JsonSchemaError, match="properties"):
        parameter_schema_spec_to_json_schema(bad_nested_props)

    prop_non_map_in_tree: dict[str, Any] = {
        "nested": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"x": 123},
        }
    }
    with pytest.raises(JsonSchemaError, match="schema specification object"):
        parameter_schema_spec_to_json_schema(prop_non_map_in_tree)

    with pytest.raises(TypeError, match="mapping"):
        define_tool(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="string name and description"):
        define_tool({"name": 123, "description": "d"})
    with pytest.raises(ValueError, match="must declare output"):
        define_tool({"name": "a", "description": "d"})
    with pytest.raises(ValueError, match="callable render"):
        define_tool(
            {"name": "a", "description": "d", "output": {"schema": {"type": "string"}}}
        )
    with pytest.raises(ValueError, match="callable execute"):
        define_tool(
            {
                "name": "a",
                "description": "d",
                "output": {"schema": {"type": "string"}, "render": lambda _a, v: []},
                "execute": "nope",
            }
        )
    with pytest.raises(ValueError, match="spec mapping"):
        define_tool(name="only-name")

    def _proj(_ex: Any, _res: Any) -> list[dict[str, str]]:
        return [{"type": "text", "text": "projected"}]

    kwarg_parity = define_tool(
        name="kw",
        description="kwarg parity",
        parameters={"x": {"type": "string", "required": True}},
        output={
            "schema": {"type": "string"},
            "render": lambda _a, v: [{"type": "text", "text": v}],
        },
        projectContent=_proj,
        presentCall=lambda args: {"card": "generic", "title": args["x"]},
        presentResult=lambda args, _r: {
            "card": "generic",
            "title": args["x"],
        },
        execute=lambda args: args["x"],
    )
    assert kwarg_parity["execute"]({"x": "hi"}, {}) == "hi"
    assert kwarg_parity["projectContent"] is _proj
    assert kwarg_parity["presentCall"]({"x": "hi"}) == {
        "card": "generic",
        "title": "hi",
    }
    assert kwarg_parity["presentResult"](
        {"x": "hi"}, {"content": [], "isError": False}
    ) == {
        "card": "generic",
        "title": "hi",
    }

    # Builtin / uninspectable execute still builds and delegates cleanly
    with patch(
        "magitech_harness.tools.schema.inspect.signature", side_effect=ValueError
    ):
        uninspectable = define_tool(
            {
                "name": "u",
                "description": "u",
                "parameters": {},
                "output": {"schema": {"type": "string"}, "render": lambda _a, _v: []},
                "execute": lambda _a, _c: "ok",
            }
        )
        assert uninspectable["execute"]({}, {"tag": 1}) == "ok"

    legacy_kw = define_tool(
        name="leg_kw",
        description="legacy",
        parameters={},
        execute=lambda args: "ok",
    )
    assert legacy_kw["name"] == "leg_kw"
    assert legacy_kw["output"]["schema"] == {"type": "string"}


def test_default_text_render_handles_str_and_non_str() -> None:
    """The legacy keyword shape's default output render produces text blocks."""
    from magitech_harness.tools.schema import _default_text_render

    assert _default_text_render({}, "hello") == [{"type": "text", "text": "hello"}]
    assert _default_text_render({}, 42) == [{"type": "text", "text": "42"}]
