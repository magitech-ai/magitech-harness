"""Enforced raw JSON Schema subset validation and iterative value checker."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from magitech_harness.llm.error import HarnessError
from magitech_harness.llm.values import (
    FrozenDict,
    FrozenList,
    assert_never,
    is_json_value,
)

JsonSchemaNode = dict[str, Any]
ObjectJsonSchema = dict[str, Any]

SCALAR_SCHEMA_TYPES = frozenset({"string", "number", "integer", "boolean", "null"})
VALID_JSON_SCHEMA_TYPES = frozenset({"object", "array", *SCALAR_SCHEMA_TYPES})
JSON_SCHEMA_ANNOTATION_KEYS = frozenset({"description", "title", "default", "examples"})
_STRUCTURAL_KEYS = (
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
)
KNOWN_JSON_SCHEMA_KEYS = frozenset(
    {"type", "oneOf", *_STRUCTURAL_KEYS, *JSON_SCHEMA_ANNOTATION_KEYS}
)


class JsonSchemaError(HarnessError):
    """Raised when a schema falls outside the supported JSON Schema subset."""

    def __init__(self, violations: Sequence[str]) -> None:
        self.violations: list[str] = list(violations)
        super().__init__(
            f"unsupported JSON schema: {'; '.join(self.violations)}",
            "UNSUPPORTED_SCHEMA",
        )


def _is_plain_mapping(value: Any) -> bool:
    return type(value) is dict or isinstance(value, FrozenDict)


def _is_plain_list(value: Any) -> bool:
    return type(value) is list or isinstance(value, FrozenList)


def _scalar_equal(left: Any, right: Any) -> bool:
    """Compare JSON scalars without conflating bool with int."""
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    return bool(left == right)


def _matches_scalar_type(type_name: str, value: Any) -> bool:
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not is_json_value(value):
        return False
    if type_name == "number":
        return True
    return isinstance(value, int) or value.is_integer()


def _validate_scalar_node_schema(
    node: Mapping[str, Any],
    type_name: str,
    node_path: str,
    violations: list[str],
) -> None:
    has_valid_enum = False
    enum_vals: list[Any] = []
    if "enum" in node:
        raw_enum = node["enum"]
        if (
            _is_plain_list(raw_enum)
            and len(raw_enum) > 0
            and all(_matches_scalar_type(type_name, item) for item in raw_enum)
        ):
            has_valid_enum = True
            enum_vals = list(raw_enum)
        else:
            violations.append(
                f"{node_path}.enum must be a non-empty array of {type_name} values"
            )

    has_valid_const = False
    if "const" in node:
        raw_const = node["const"]
        if _matches_scalar_type(type_name, raw_const):
            has_valid_const = True
        else:
            violations.append(f"{node_path}.const must be a {type_name} value")

    if has_valid_enum and has_valid_const:
        raw_const = node["const"]
        if not any(_scalar_equal(raw_const, item) for item in enum_vals):
            violations.append(
                f"{node_path}.const must be one of {node_path}.enum when both are declared"
            )


def _validate_object_node_schema(
    node: Mapping[str, Any],
    node_path: str,
    violations: list[str],
    stack: list[tuple[str, str, Any]],
) -> None:
    prop_names: set[str] | None = None
    if "properties" in node:
        raw_props = node["properties"]
        if _is_plain_mapping(raw_props):
            prop_names = {k for k in raw_props if isinstance(k, str)}
            for k, prop_schema in reversed(list(raw_props.items())):
                if isinstance(k, str):
                    stack.append(("enter", f"{node_path}.properties.{k}", prop_schema))
                else:
                    violations.append(f"{node_path}.properties must use string keys")
        else:
            violations.append(f"{node_path}.properties must be an object of schemas")

    if "required" in node:
        raw_req = node["required"]
        if _is_plain_list(raw_req) and all(isinstance(r, str) for r in raw_req):
            known = prop_names if prop_names is not None else set()
            for req in raw_req:
                if req not in known:
                    violations.append(
                        f'{node_path}.required names "{req}" which is not in properties'
                    )
        else:
            violations.append(f"{node_path}.required must be an array of strings")

    if "additionalProperties" in node and not isinstance(
        node["additionalProperties"], bool
    ):
        violations.append(f"{node_path}.additionalProperties must be a boolean")


def _validate_annotations(
    node: Mapping[str, Any], node_path: str, violations: list[str]
) -> None:
    if "description" in node and not isinstance(node["description"], str):
        violations.append(f"{node_path}.description must be a string")
    if "title" in node and not isinstance(node["title"], str):
        violations.append(f"{node_path}.title must be a string")
    if "default" in node and not is_json_value(node["default"]):
        violations.append(f"{node_path}.default annotation must be lossless JSON data")
    if "examples" in node and not is_json_value(node["examples"]):
        violations.append(f"{node_path}.examples annotation must be lossless JSON data")


def assert_supported_json_schema(schema: Any, *, path: str = "schema") -> None:
    """Validate iteratively that *schema* stays within the supported JSON Schema subset."""
    violations: list[str] = []
    active_ids: set[int] = set()
    stack: list[tuple[str, str, Any]] = [("enter", path, schema)]

    while stack:
        op, node_path, payload = stack.pop()
        if op == "exit":
            active_ids.remove(payload)
            continue

        if not _is_plain_mapping(payload) or not all(
            isinstance(k, str) for k in payload
        ):
            violations.append(f"{node_path} must be a schema object")
            continue

        node_id = id(payload)
        if node_id in active_ids:
            violations.append(f"{node_path} is circular")
            continue

        active_ids.add(node_id)
        stack.append(("exit", "", node_id))

        for key in payload:
            if key not in KNOWN_JSON_SCHEMA_KEYS:
                violations.append(f"{node_path}.{key} is not a supported keyword")

        _validate_annotations(payload, node_path, violations)

        has_type = "type" in payload
        has_one_of = "oneOf" in payload
        if has_type and has_one_of:
            violations.append(f"{node_path} cannot declare both type and oneOf")
            continue

        if has_one_of:
            for key in _STRUCTURAL_KEYS:
                if key in payload:
                    violations.append(
                        f"{node_path}.{key} is not supported beside oneOf"
                    )
            branches = payload["oneOf"]
            if not _is_plain_list(branches) or len(branches) < 2:
                violations.append(
                    f"{node_path}.oneOf must be an array of at least two schemas"
                )
            else:
                for idx in range(len(branches) - 1, -1, -1):
                    stack.append(("enter", f"{node_path}.oneOf[{idx}]", branches[idx]))
            continue

        if not has_type:
            for key in _STRUCTURAL_KEYS:
                if key in payload:
                    violations.append(f"{node_path}.{key} requires type or oneOf")
            continue

        type_val = payload["type"]
        if _is_plain_list(type_val):
            violations.append(
                f"{node_path}.type must be a single type string "
                "(type arrays are not supported)"
            )
            continue
        if not isinstance(type_val, str) or type_val not in VALID_JSON_SCHEMA_TYPES:
            allowed = ", ".join(sorted(VALID_JSON_SCHEMA_TYPES))
            violations.append(f"{node_path}.type must be one of: {allowed}")
            continue

        if type_val != "array" and "items" in payload:
            violations.append(
                f'{node_path}.items is not supported on type "{type_val}"'
            )
        if type_val != "object":
            for key in ("properties", "required", "additionalProperties"):
                if key in payload:
                    violations.append(
                        f'{node_path}.{key} is not supported on type "{type_val}"'
                    )
        if type_val not in SCALAR_SCHEMA_TYPES:
            for key in ("enum", "const"):
                if key in payload:
                    violations.append(
                        f'{node_path}.{key} is not supported on type "{type_val}"'
                    )

        if type_val == "object":
            _validate_object_node_schema(payload, node_path, violations, stack)
        elif type_val == "array":
            if "items" in payload:
                stack.append(("enter", f"{node_path}.items", payload["items"]))
        else:
            _validate_scalar_node_schema(payload, type_val, node_path, violations)

    if violations:
        raise JsonSchemaError(violations)


def assert_object_json_schema(schema: Any, *, path: str = "schema") -> None:
    """Assert *schema* is valid and has an explicit root ``type: 'object'``."""
    assert_supported_json_schema(schema, path=path)
    if schema.get("type") != "object":
        raise JsonSchemaError(
            [f'{path}.type must be "object" (structured output is object-rooted)']
        )


def _diagnostic_path(path: str) -> str:
    return "arguments" if path == "" else path


def _format_child_path(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _is_finite_json_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and not (
            isinstance(value, float) and value == 0.0 and math.copysign(1.0, value) < 0
        )
    )


def _validate_scalar_value(
    node: Mapping[str, Any],
    type_name: str,
    value: Any,
    val_path: str,
    violations: list[str],
) -> None:
    if type_name == "string":
        if not isinstance(value, str):
            violations.append(f'"{val_path}" must be a string')
            return
    elif type_name == "boolean":
        if not isinstance(value, bool):
            violations.append(f'"{val_path}" must be a boolean')
            return
    elif type_name == "null":
        if value is not None:
            violations.append(f'"{val_path}" must be null')
            return
    elif type_name == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            violations.append(f'"{val_path}" must be a number')
            return
        if not _is_finite_json_number(value):
            violations.append(f'"{val_path}" must be a finite JSON number')
            return
    else:  # type_name == "integer"
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not _is_finite_json_number(value)
            or (isinstance(value, float) and not value.is_integer())
        ):
            violations.append(f'"{val_path}" must be an integer')
            return

    if "enum" in node and _is_plain_list(node["enum"]):
        enum_vals = node["enum"]
        if not any(_scalar_equal(value, item) for item in enum_vals):
            encoded = json.dumps(list(enum_vals), separators=(",", ":"))
            violations.append(f'"{val_path}" must be one of {encoded}')
            return
    if "const" in node and not _scalar_equal(value, node["const"]):
        encoded = json.dumps(node["const"])
        violations.append(f'"{val_path}" must be {encoded}')


def _validate_object_value(
    node: Mapping[str, Any],
    value: Any,
    val_path: str,
    prop_prefix: str,
    violations: list[str],
    stack: list[tuple[Any, ...]],
) -> None:
    if not _is_plain_mapping(value):
        violations.append(f'"{val_path}" must be an object')
        return

    props = node.get("properties")
    declared: Mapping[str, Any] = (
        cast(Mapping[str, Any], props) if _is_plain_mapping(props) else {}
    )
    required = node.get("required")
    required_list: Sequence[Any] = (
        cast(Sequence[Any], required) if _is_plain_list(required) else ()
    )
    for req_key in required_list:
        if req_key not in value:
            child_path = _format_child_path(prop_prefix, req_key)
            violations.append(f'missing required property "{child_path}"')

    closed = node.get("additionalProperties") is False
    has_lossy_unvalidated = False
    pending_checks: list[tuple[Any, ...]] = []
    for k, v in value.items():
        if not isinstance(k, str):
            has_lossy_unvalidated = True
            continue
        child_path = _format_child_path(prop_prefix, k)
        if k in declared:
            pending_checks.append(
                ("check", declared[k], v, child_path, child_path, violations)
            )
        elif closed:
            violations.append(
                f'"{child_path}" is not a declared property (additionalProperties: false)'
            )
        elif not is_json_value(v):
            has_lossy_unvalidated = True

    if has_lossy_unvalidated:
        violations.append(f'"{val_path}" must be a lossless JSON object')
    stack.extend(reversed(pending_checks))


def validate_json_schema_value(
    schema: Mapping[str, Any],
    value: Any,
    *,
    name: str = "value",
    omit_root_name_on_properties: bool | None = None,
) -> list[str]:
    """Iteratively validate *value* against *schema*, returning human violation messages."""
    violations: list[str] = []
    omit_root = (
        name == ""
        if omit_root_name_on_properties is None
        else omit_root_name_on_properties
    )
    root_val_path = _diagnostic_path(name)
    root_prefix = "" if omit_root else name
    stack: list[tuple[Any, ...]] = [
        ("check", schema, value, root_val_path, root_prefix, violations)
    ]

    while stack:
        frame = stack.pop()
        kind = frame[0]
        if kind == "oneof_next":
            (
                _,
                branches,
                index,
                matched,
                val,
                val_path,
                prop_prefix,
                prev_bv,
                target,
            ) = frame
            if index > 0 and prev_bv is not None and len(prev_bv) == 0:
                matched += 1
            if index < len(branches):
                next_bv: list[str] = []
                stack.append(
                    (
                        "oneof_next",
                        branches,
                        index + 1,
                        matched,
                        val,
                        val_path,
                        prop_prefix,
                        next_bv,
                        target,
                    )
                )
                stack.append(
                    ("check", branches[index], val, val_path, prop_prefix, next_bv)
                )
            elif matched != 1:
                target.append(
                    f'"{val_path}" must match exactly one oneOf branch (matched {matched})'
                )
            continue

        _, node, val, val_path, prop_prefix, target = frame
        if "oneOf" in node:
            branches = node["oneOf"]
            stack.append(
                (
                    "oneof_next",
                    branches,
                    0,
                    0,
                    val,
                    val_path,
                    prop_prefix,
                    None,
                    target,
                )
            )
            continue

        if "type" not in node:
            if not is_json_value(val):
                target.append(f'"{val_path}" must be a lossless JSON value')
            continue

        type_val = node["type"]
        if type_val == "object":
            _validate_object_value(node, val, val_path, prop_prefix, target, stack)
        elif type_val == "array":
            if not _is_plain_list(val):
                target.append(f'"{val_path}" must be an array')
                continue
            if "items" in node:
                item_schema = node["items"]
                for idx in range(len(val) - 1, -1, -1):
                    item_path = f"{prop_prefix}[{idx}]"
                    stack.append(
                        ("check", item_schema, val[idx], item_path, item_path, target)
                    )
            elif not is_json_value(val):
                target.append(f'"{val_path}" must be a dense lossless JSON array')
        elif type_val in SCALAR_SCHEMA_TYPES:
            _validate_scalar_value(node, type_val, val, val_path, target)
        else:
            assert_never(type_val, f"schema node at {val_path}")

    return violations


__all__ = [
    "JSON_SCHEMA_ANNOTATION_KEYS",
    "KNOWN_JSON_SCHEMA_KEYS",
    "SCALAR_SCHEMA_TYPES",
    "VALID_JSON_SCHEMA_TYPES",
    "JsonSchemaError",
    "JsonSchemaNode",
    "ObjectJsonSchema",
    "assert_object_json_schema",
    "assert_supported_json_schema",
    "validate_json_schema_value",
]
