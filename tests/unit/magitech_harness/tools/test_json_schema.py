"""Unit tests for the enforced raw JSON Schema subset."""

from __future__ import annotations

import pytest

from magitech_harness.llm.error import HarnessError
from magitech_harness.tools.json_schema import (
    JsonSchemaError,
    assert_object_json_schema,
    assert_supported_json_schema,
    validate_json_schema_value,
)


def _violations(schema: object, *, object_root: bool = False) -> list[str]:
    """Return JsonSchemaError violations for a schema under test."""
    with pytest.raises(JsonSchemaError) as exc_info:
        if object_root:
            assert_object_json_schema(schema)
        else:
            assert_supported_json_schema(schema)
    assert isinstance(exc_info.value, HarnessError)
    assert exc_info.value.code == "UNSUPPORTED_SCHEMA"
    return exc_info.value.violations


def test_accepts_every_supported_node() -> None:
    """Every JSON root and supported structural/scalar schema passes."""
    for schema in (
        {"type": "string"},
        {"type": "number"},
        {"type": "integer"},
        {"type": "boolean"},
        {"type": "null"},
        {"type": "array", "items": {"type": "string"}},
        {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                "free": {},
            },
            "required": ["nested"],
            "additionalProperties": True,
        },
        {"oneOf": [{"type": "string"}, {"type": "number"}]},
        {
            "description": "any JSON",
            "title": "JSON",
            "default": None,
            "examples": [1, "x"],
        },
    ):
        assert_supported_json_schema(schema)


def test_object_root_guard() -> None:
    """assert_object_json_schema requires root type == 'object'."""
    assert_object_json_schema({"type": "object"})
    for schema in (
        {},
        {"type": "string"},
        {"type": "array"},
        {"oneOf": [{"type": "string"}, {"type": "null"}]},
    ):
        assert _violations(schema, object_root=True) == [
            'schema.type must be "object" (structured output is object-rooted)'
        ]


def test_rejects_non_schema_unknown_type_and_type_arrays() -> None:
    """Rejects non-mapping roots, unknown type strings, and type lists."""
    assert _violations(None) == ["schema must be a schema object"]
    assert _violations([]) == ["schema must be a schema object"]
    assert _violations("no") == ["schema must be a schema object"]
    assert _violations({"type": "tuple"})[0].startswith("schema.type must be one of")
    assert _violations({"type": ["string", "null"]}) == [
        "schema.type must be a single type string (type arrays are not supported)"
    ]


def test_enforces_oneof_vocabulary_and_branch_count() -> None:
    """Enforces >=2 branches, no type+oneOf, and no sibling keywords."""
    for bad in ([], [{}], "x"):
        assert _violations({"oneOf": bad}) == [
            "schema.oneOf must be an array of at least two schemas"
        ]
    assert _violations({"type": "string", "oneOf": [{}, {}]}) == [
        "schema cannot declare both type and oneOf"
    ]
    assert _violations(
        {"oneOf": [{"type": "string"}, {"type": "number"}], "items": {}}
    ) == ["schema.items is not supported beside oneOf"]
    assert (
        "schema.oneOf[1].type"
        in _violations({"oneOf": [{"type": "string"}, {"type": "weird"}]})[0]
    )


def test_rejects_unknown_and_misplaced_keywords() -> None:
    """Reports unknown keywords, misplaced keywords, and missing combinators."""
    for kw in ("anyOf", "allOf", "not", "pattern", "minimum", "maxLength", "$ref"):
        assert (
            f"schema.{kw} is not a supported keyword"
            in _violations({"type": "object", kw: []})[0]
        )
    assert _violations({"type": "object", "items": {}}) == [
        'schema.items is not supported on type "object"'
    ]
    assert _violations({"type": "array", "properties": {}}) == [
        'schema.properties is not supported on type "array"'
    ]
    assert _violations({"type": "object", "enum": ["x"]}) == [
        'schema.enum is not supported on type "object"'
    ]
    assert _violations({"type": "array", "const": None}) == [
        'schema.const is not supported on type "array"'
    ]
    assert _violations(
        {
            "properties": {},
            "required": [],
            "additionalProperties": True,
            "items": {},
            "enum": [],
            "const": None,
        }
    ) == [
        "schema.properties requires type or oneOf",
        "schema.required requires type or oneOf",
        "schema.additionalProperties requires type or oneOf",
        "schema.items requires type or oneOf",
        "schema.enum requires type or oneOf",
        "schema.const requires type or oneOf",
    ]


def test_reports_every_independent_violation() -> None:
    """Collects multiple independent violations in a single pass."""
    assert (
        len(
            _violations(
                {
                    "type": "object",
                    "pattern": "x",
                    "properties": {
                        "a": {"type": "weird"},
                        "b": {"type": "string", "minimum": 1},
                    },
                }
            )
        )
        == 3
    )


def test_validates_object_properties_required_and_openness() -> None:
    """Validates properties mapping, required string list subset, and openness bool."""
    assert _violations({"type": "object", "properties": []}) == [
        "schema.properties must be an object of schemas"
    ]
    assert _violations({"type": "object", "properties": {"a": "x"}}) == [
        "schema.properties.a must be a schema object"
    ]
    assert _violations({"type": "object", "required": "a"}) == [
        "schema.required must be an array of strings"
    ]
    assert _violations({"type": "object", "required": [1]}) == [
        "schema.required must be an array of strings"
    ]
    assert _violations(
        {"type": "object", "properties": {}, "required": ["missing"]}
    ) == ['schema.required names "missing" which is not in properties']
    assert _violations({"type": "object", "additionalProperties": "yes"}) == [
        "schema.additionalProperties must be a boolean"
    ]


def test_requires_type_correct_scalar_enum_and_const() -> None:
    """Scalar enum/const match type accurately, distinguishing bool from int."""
    for valid in (
        {"type": "string", "enum": ["a"], "const": "a"},
        {"type": "number", "enum": [1.5], "const": 1.5},
        {"type": "integer", "enum": [1], "const": 1},
        {"type": "boolean", "enum": [True], "const": True},
        {"type": "null", "enum": [None], "const": None},
    ):
        assert_supported_json_schema(valid)

    assert _violations({"type": "string", "enum": []}) == [
        "schema.enum must be a non-empty array of string values"
    ]
    assert _violations({"type": "number", "enum": ["1"]}) == [
        "schema.enum must be a non-empty array of number values"
    ]
    assert _violations({"type": "integer", "enum": [1.5]}) == [
        "schema.enum must be a non-empty array of integer values"
    ]
    assert _violations({"type": "integer", "enum": [True]}) == [
        "schema.enum must be a non-empty array of integer values"
    ]
    assert _violations({"type": "number", "enum": [float("nan")]}) == [
        "schema.enum must be a non-empty array of number values"
    ]
    assert _violations({"type": "number", "const": -0.0}) == [
        "schema.const must be a number value"
    ]
    assert _violations({"type": "boolean", "const": 1}) == [
        "schema.const must be a boolean value"
    ]
    assert _violations({"type": "integer", "enum": [1], "const": True}) == [
        "schema.const must be a integer value"
    ]
    assert _violations({"type": "string", "enum": ["a"], "const": "b"}) == [
        "schema.const must be one of schema.enum when both are declared"
    ]


def test_validates_annotations_and_lossless_payloads() -> None:
    """Validates description, title, default, examples."""
    assert _violations({"description": 1}) == ["schema.description must be a string"]
    assert _violations({"title": 1}) == ["schema.title must be a string"]
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    for bad in (float("inf"), (1,), cyclic, object()):
        assert _violations({"default": bad}) == [
            "schema.default annotation must be lossless JSON data"
        ]
        assert _violations({"examples": bad}) == [
            "schema.examples annotation must be lossless JSON data"
        ]


def test_rejects_circular_schemas_but_permits_dag_reuse() -> None:
    """Cycles fail with circular error while shared sibling nodes succeed."""
    cyclic: dict[str, object] = {"type": "object"}
    cyclic["properties"] = {"self": cyclic}
    assert _violations(cyclic) == ["schema.properties.self is circular"]
    leaf = {"type": "string"}
    assert_supported_json_schema(
        {"type": "object", "properties": {"a": leaf, "b": leaf}}
    )


def test_deep_unions_validate_iteratively_without_stack_overflow() -> None:
    """5000-deep oneOf validates iteratively without hitting Python recursion limits."""
    schema: dict[str, object] = {"type": "string"}
    for _ in range(5000):
        schema = {"oneOf": [schema, {"type": "null"}]}
    assert_supported_json_schema(schema)
    assert validate_json_schema_value(schema, "leaf") == []
    assert validate_json_schema_value(schema, 42) == [
        '"value" must match exactly one oneOf branch (matched 0)'
    ]


def test_validate_json_schema_value_scalars_enums_and_objects() -> None:
    """Validates scalar types, enums, consts, required fields, openness, arrays."""
    assert validate_json_schema_value({"type": "string"}, "x") == []
    assert validate_json_schema_value({"type": "number"}, 1.5) == []
    assert validate_json_schema_value({"type": "integer"}, 2) == []
    assert validate_json_schema_value({"type": "boolean"}, True) == []
    assert validate_json_schema_value({"type": "null"}, None) == []
    assert validate_json_schema_value({"type": "string"}, 1) == [
        '"value" must be a string'
    ]
    assert validate_json_schema_value({"type": "number"}, "1") == [
        '"value" must be a number'
    ]
    assert validate_json_schema_value({"type": "number"}, True) == [
        '"value" must be a number'
    ]
    assert validate_json_schema_value({"type": "number"}, float("nan")) == [
        '"value" must be a finite JSON number'
    ]
    assert validate_json_schema_value({"type": "number"}, -0.0) == [
        '"value" must be a finite JSON number'
    ]
    assert validate_json_schema_value({"type": "integer"}, 1.5) == [
        '"value" must be an integer'
    ]
    assert validate_json_schema_value({"type": "integer"}, False) == [
        '"value" must be an integer'
    ]
    assert validate_json_schema_value({"type": "boolean"}, "true") == [
        '"value" must be a boolean'
    ]
    assert validate_json_schema_value({"type": "null"}, 0) == ['"value" must be null']

    constrained = {"type": "string", "enum": ["a", "b"], "const": "a"}
    assert validate_json_schema_value(constrained, "a") == []
    assert validate_json_schema_value(constrained, "c") == [
        '"value" must be one of ["a","b"]'
    ]
    assert validate_json_schema_value(constrained, "b") == ['"value" must be "a"']

    obj_schema = {
        "type": "object",
        "properties": {
            "file": {"type": "string"},
            "nested": {
                "type": "object",
                "properties": {"line": {"type": "integer"}},
                "required": ["line"],
                "additionalProperties": False,
            },
        },
        "required": ["file"],
    }
    assert (
        validate_json_schema_value(
            obj_schema, {"file": "a", "extra": [1], "nested": {"line": 2}}
        )
        == []
    )
    assert validate_json_schema_value(obj_schema, {"nested": {"line": 1}}) == [
        'missing required property "value.file"'
    ]
    assert validate_json_schema_value(obj_schema, {"file": 1, "nested": {}}) == [
        '"value.file" must be a string',
        'missing required property "value.nested.line"',
    ]
    assert validate_json_schema_value(
        obj_schema, {"file": "a", "nested": {"line": 1, "extra": True}}
    ) == [
        '"value.nested.extra" is not a declared property (additionalProperties: false)'
    ]
    assert validate_json_schema_value(obj_schema, "x") == ['"value" must be an object']
    assert validate_json_schema_value({"type": "object"}, {"bad": float("inf")}) == [
        '"value" must be a lossless JSON object'
    ]


def test_validate_json_schema_value_arrays_oneof_and_unconstrained() -> None:
    """Validates dense arrays, exact-one oneOf overlap, and unconstrained nodes."""
    arr_schema = {"type": "array", "items": {"type": "integer"}}
    assert validate_json_schema_value(arr_schema, [1, 2]) == []
    assert validate_json_schema_value(arr_schema, [1, 1.5]) == [
        '"value[1]" must be an integer'
    ]
    assert validate_json_schema_value(arr_schema, "x") == ['"value" must be an array']
    assert validate_json_schema_value({"type": "array"}, [1, float("nan")]) == [
        '"value" must be a dense lossless JSON array'
    ]

    disjoint = {"oneOf": [{"type": "string"}, {"type": "number"}]}
    assert validate_json_schema_value(disjoint, "x") == []
    assert validate_json_schema_value(disjoint, None) == [
        '"value" must match exactly one oneOf branch (matched 0)'
    ]
    overlap = {"oneOf": [{"type": "number"}, {"type": "integer"}]}
    assert validate_json_schema_value(overlap, 1) == [
        '"value" must match exactly one oneOf branch (matched 2)'
    ]
    assert validate_json_schema_value(overlap, 1.5) == []

    for valid in (None, True, 1, "x", [1], {"x": None}):
        assert validate_json_schema_value({}, valid) == []
    for bad in (float("inf"), -0.0, object(), (1,)):
        assert validate_json_schema_value({}, bad) == [
            '"value" must be a lossless JSON value'
        ]


def test_validate_json_schema_value_forged_schema_assert_never() -> None:
    """Unknown type node hitting value validator raises AssertionError/ValueError."""
    with pytest.raises((AssertionError, ValueError), match="tuple"):
        validate_json_schema_value({"type": "tuple"}, 1)


def test_json_schema_edge_branches() -> None:
    """Covers non-string property keys, bool/int scalar isolation, NaN integer, array pass."""
    assert _violations({"type": "object", "properties": {1: {"type": "string"}}}) == [
        "schema.properties must use string keys"
    ]
    assert validate_json_schema_value({"type": "integer"}, float("nan")) == [
        '"value" must be an integer'
    ]
    assert validate_json_schema_value({"type": "integer"}, -0.0) == [
        '"value" must be an integer'
    ]
    assert validate_json_schema_value({"type": "integer"}, float("inf")) == [
        '"value" must be an integer'
    ]
    assert validate_json_schema_value(
        {"type": "array", "items": {"type": "string"}}, [1], name=""
    ) == ['"[0]" must be a string']
    assert validate_json_schema_value({"type": "number", "enum": [True]}, 1) == [
        '"value" must be one of [true]'
    ]
    assert validate_json_schema_value({"type": "object"}, {1: "a"}) == [
        '"value" must be a lossless JSON object'
    ]
    assert validate_json_schema_value({"type": "array"}, [1, "a", True]) == []
