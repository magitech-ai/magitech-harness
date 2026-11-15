"""Unified author schema DSL compiler, argument validation, and define_tool."""

from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, cast

from magitech_harness.llm.error import HarnessError
from magitech_harness.llm.types import ToolSchema
from magitech_harness.llm.values import FrozenDict, FrozenList, snapshot_json_value
from magitech_harness.tools.definition import Tool, legacy_define_tool
from magitech_harness.tools.json_schema import (
    JSON_SCHEMA_ANNOTATION_KEYS,
    SCALAR_SCHEMA_TYPES,
    JsonSchemaError,
    assert_supported_json_schema,
    validate_json_schema_value,
)

_COMMON_DSL_KEYS = frozenset(JSON_SCHEMA_ANNOTATION_KEYS)
_OBJECT_DSL_KEYS = frozenset(
    {"type", "properties", "additionalProperties", *_COMMON_DSL_KEYS}
)
_ARRAY_DSL_KEYS = frozenset({"type", "items", *_COMMON_DSL_KEYS})
_SCALAR_DSL_KEYS = frozenset({"type", "enum", "const", *_COMMON_DSL_KEYS})
_JSON_DSL_KEYS = frozenset({"type", *_COMMON_DSL_KEYS})
_ONEOF_DSL_KEYS = frozenset({"oneOf", *_COMMON_DSL_KEYS})


class ToolArgsError(HarnessError):
    """Raised when tool arguments violate the declared parameter schema."""

    def __init__(self, violations: Sequence[str]) -> None:
        self.violations: list[str] = list(violations)
        super().__init__(
            f"invalid arguments: {'; '.join(self.violations)}",
            "INVALID_ARGS",
        )


def _is_plain_mapping(value: Any) -> bool:
    return type(value) is dict or isinstance(value, FrozenDict)


def _is_plain_list(value: Any) -> bool:
    return type(value) is list or isinstance(value, FrozenList)


def _allowed_keys_for_spec(
    node: Mapping[str, Any], *, allow_required: bool
) -> frozenset[str]:
    if "oneOf" in node:
        allowed = _ONEOF_DSL_KEYS | {"type"}
    elif node.get("type") == "object":
        allowed = _OBJECT_DSL_KEYS
    elif node.get("type") == "array":
        allowed = _ARRAY_DSL_KEYS
    elif node.get("type") == "json":
        allowed = _JSON_DSL_KEYS
    elif node.get("type") in SCALAR_SCHEMA_TYPES:
        allowed = _SCALAR_DSL_KEYS
    else:
        return frozenset()
    return allowed | {"required"} if allow_required else allowed


def _snapshot_or_raw(raw: Any) -> Any:
    snapped = snapshot_json_value(raw)
    return raw if (raw is not None and snapped is None) else snapped


def _copy_annotations(src: Mapping[str, Any], dst: dict[str, Any]) -> None:
    for key in JSON_SCHEMA_ANNOTATION_KEYS:
        if key in src:
            dst[key] = _snapshot_or_raw(src[key])


def _compile_dsl_node(
    root: Any,
    *,
    root_path: str,
) -> dict[str, Any]:
    """Iteratively compile a ValueSchemaSpec node into a validated JSON Schema dict."""
    active_ids: set[int] = set()
    holder: dict[str, Any] = {}

    def _assign_root(compiled: dict[str, Any]) -> None:
        holder["root"] = compiled

    stack: list[tuple[Any, ...]] = [("enter", root, root_path, False, _assign_root)]
    while stack:
        frame = stack.pop()
        if frame[0] == "exit":
            active_ids.remove(frame[1])
            continue

        _, node, node_path, allow_required, setter = frame
        if not _is_plain_mapping(node) or not all(isinstance(k, str) for k in node):
            raise JsonSchemaError(
                [f"{node_path} must be a schema specification object"]
            )

        node_id = id(node)
        if node_id in active_ids:
            raise JsonSchemaError([f"{node_path} is circular"])

        if allow_required and "required" in node and node["required"] is not True:
            raise JsonSchemaError(
                [f"{node_path}.required must be omitted or strictly true"]
            )
        if not allow_required and "required" in node:
            raise JsonSchemaError([f"{node_path}.required is only valid on properties"])

        allowed = _allowed_keys_for_spec(node, allow_required=allow_required)
        if not allowed:
            raise JsonSchemaError([f"{node_path} must declare a valid type or oneOf"])

        extra_keys = [k for k in node if k not in allowed]
        if extra_keys:
            raise JsonSchemaError(
                [
                    f"{node_path}.{k} is not a supported schema DSL keyword"
                    for k in extra_keys
                ]
            )
        if "oneOf" in node and "type" in node:
            raise JsonSchemaError([f"{node_path} cannot declare both type and oneOf"])

        active_ids.add(node_id)
        stack.append(("exit", node_id))

        out: dict[str, Any] = {}
        _copy_annotations(node, out)

        if "oneOf" in node:
            branches = node["oneOf"]
            if not _is_plain_list(branches) or len(branches) < 2:
                raise JsonSchemaError(
                    [f"{node_path}.oneOf must be an array of at least two schemas"]
                )
            compiled_branches: list[Any] = [None] * len(branches)
            out["oneOf"] = compiled_branches
            setter(out)
            for idx in range(len(branches) - 1, -1, -1):

                def _set_branch(
                    val: dict[str, Any],
                    target: list[Any] = compiled_branches,
                    pos: int = idx,
                ) -> None:
                    target[pos] = val

                stack.append(
                    (
                        "enter",
                        branches[idx],
                        f"{node_path}.oneOf[{idx}]",
                        False,
                        _set_branch,
                    )
                )
            continue

        type_val = node["type"]
        if type_val == "json":
            setter(out)
            continue

        out["type"] = type_val
        if type_val == "object":
            if "additionalProperties" not in node or not isinstance(
                node["additionalProperties"], bool
            ):
                raise JsonSchemaError(
                    [f"{node_path}.additionalProperties must be declared as a boolean"]
                )
            out["additionalProperties"] = node["additionalProperties"]
            if "properties" in node:
                raw_props = node["properties"]
                if not _is_plain_mapping(raw_props) or not all(
                    isinstance(k, str) for k in raw_props
                ):
                    raise JsonSchemaError(
                        [
                            f"{node_path}.properties must be an object mapping property names"
                        ]
                    )
                compiled_props: dict[str, Any] = {}
                required_keys: list[str] = []
                out["properties"] = compiled_props
                for prop_name, prop_spec in raw_props.items():
                    if (
                        _is_plain_mapping(prop_spec)
                        and prop_spec.get("required") is True
                    ):
                        required_keys.append(prop_name)
                if required_keys:
                    out["required"] = required_keys
                setter(out)
                for prop_name, prop_spec in reversed(list(raw_props.items())):

                    def _set_prop(
                        val: dict[str, Any],
                        target: dict[str, Any] = compiled_props,
                        key: str = prop_name,
                    ) -> None:
                        target[key] = val

                    stack.append(
                        (
                            "enter",
                            prop_spec,
                            f"{node_path}.properties.{prop_name}",
                            True,
                            _set_prop,
                        )
                    )
            else:
                setter(out)
            continue

        if type_val == "array":
            setter(out)
            if "items" in node:

                def _set_items(
                    val: dict[str, Any], target: dict[str, Any] = out
                ) -> None:
                    target["items"] = val

                stack.append(
                    ("enter", node["items"], f"{node_path}.items", False, _set_items)
                )
            continue

        if "enum" in node:
            out["enum"] = (
                _snapshot_or_raw(list(node["enum"]))
                if _is_plain_list(node["enum"])
                else node["enum"]
            )
        if "const" in node:
            out["const"] = _snapshot_or_raw(node["const"])
        setter(out)

    compiled_root = holder["root"]
    assert_supported_json_schema(compiled_root, path=root_path)
    return cast(dict[str, Any], compiled_root)


def value_schema_spec_to_json_schema(
    spec: Any, *, path: str = "schema"
) -> dict[str, Any]:
    """Compile a ValueSchemaSpec DSL definition into a validated JSON Schema object."""
    return _compile_dsl_node(spec, root_path=path)


def parameter_schema_spec_to_json_schema(
    spec: Any, *, path: str = "parameters"
) -> dict[str, Any]:
    """Compile a ParameterSchemaSpec map into an open root object JSON Schema."""
    if not _is_plain_mapping(spec) or not all(isinstance(k, str) for k in spec):
        raise JsonSchemaError([f"{path} must be an object mapping parameter names"])

    spec_id = id(spec)
    compiled_props: dict[str, Any] = {}
    required_keys: list[str] = []
    for key, prop_spec in spec.items():
        if _is_plain_mapping(prop_spec) and prop_spec.get("required") is True:
            required_keys.append(key)
        compiled_props[key] = _compile_property_node(
            prop_spec, root_ids={spec_id}, prop_path=f"{path}.{key}"
        )

    result: dict[str, Any] = {
        "type": "object",
        "properties": compiled_props,
    }
    if required_keys:
        result["required"] = required_keys
    assert_supported_json_schema(result, path=path)
    return result


def _compile_property_node(
    prop_spec: Any, *, root_ids: set[int], prop_path: str
) -> dict[str, Any]:
    if not _is_plain_mapping(prop_spec) or not all(
        isinstance(k, str) for k in prop_spec
    ):
        raise JsonSchemaError([f"{prop_path} must be a schema specification object"])
    if "required" in prop_spec and prop_spec["required"] is not True:
        raise JsonSchemaError(
            [f"{prop_path}.required must be omitted or strictly true"]
        )

    # Check if any nested `properties` directly aliases root `spec` before stripping `required`.
    stack: list[Any] = [prop_spec]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        if not _is_plain_mapping(cur):
            continue
        cur_id = id(cur)
        if cur_id in root_ids:
            raise JsonSchemaError([f"{prop_path} is circular"])
        if cur_id in seen:
            continue
        seen.add(cur_id)
        if "properties" in cur:
            stack.append(cur["properties"])
            if _is_plain_mapping(cur["properties"]):
                stack.extend(cur["properties"].values())
        if "items" in cur:
            stack.append(cur["items"])
        if "oneOf" in cur and _is_plain_list(cur["oneOf"]):
            stack.extend(cur["oneOf"])

    if "required" in prop_spec:
        stripped = {k: v for k, v in prop_spec.items() if k != "required"}
        return _compile_dsl_node(stripped, root_path=prop_path)
    return _compile_dsl_node(prop_spec, root_path=prop_path)


def validate_args(spec: Mapping[str, Any], args: Any) -> list[str]:
    """Validate tool call arguments against a ParameterSchemaSpec."""
    compiled = parameter_schema_spec_to_json_schema(spec)
    return validate_json_schema_value(compiled, args, name="")


def validate_value(
    spec: Mapping[str, Any], value: Any, *, path: str = "value"
) -> list[str]:
    """Validate a value against a ValueSchemaSpec."""
    compiled = value_schema_spec_to_json_schema(spec, path=path)
    return validate_json_schema_value(compiled, value, name=path)


def validate_timeout_ms(timeout_ms: Any) -> float | None:
    """Validate optional tool timeoutMs budget as a finite positive number."""
    if timeout_ms is None:
        return None
    if (
        isinstance(timeout_ms, bool)
        or not isinstance(timeout_ms, (int, float))
        or not math.isfinite(timeout_ms)
        or timeout_ms <= 0
    ):
        raise ValueError("timeoutMs must be a finite positive number")
    return float(timeout_ms)


def project_tool_schema(
    definition: Mapping[str, Any],
    detach_parameters: bool = True,
    *,
    include_defer_loading: bool = True,
) -> ToolSchema:
    """Project one ToolDefinition to the model-visible ToolSchema (dropping host callbacks)."""
    name = str(definition["name"])
    raw_params = definition["parameters"]
    if detach_parameters:
        snapped = snapshot_json_value(raw_params)
        if raw_params is not None and snapped is None:
            raise ValueError(
                f'tool "{name}" parameters must be lossless JSON before schema projection'
            )
        parameters = cast(dict[str, Any], snapped)
    else:
        parameters = cast(dict[str, Any], raw_params)
    schema: ToolSchema = {
        "name": name,
        "description": str(definition["description"]),
        "parameters": parameters,
    }
    if include_defer_loading and definition.get("deferLoading") is True:
        schema["deferLoading"] = True
    return schema


def _soft_validate_args(compiled_params: dict[str, Any], args: Any) -> bool:
    return len(validate_json_schema_value(compiled_params, args, name="")) == 0


def define_tool_definition(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build a schema-validated ToolDefinition carrying strict execute and soft UI wrappers."""
    name = spec.get("name")
    description = spec.get("description")
    if not isinstance(name, str) or not isinstance(description, str):
        raise ValueError("defineTool requires string name and description")  # noqa: TRY004
    if "output" not in spec or not isinstance(spec["output"], Mapping):
        raise ValueError(f'tool "{name}" must declare output {{ schema, render }}')

    raw_output = spec["output"]
    if "schema" not in raw_output or not callable(raw_output.get("render")):
        raise ValueError(
            f'tool "{name}" output must declare schema and callable render'
        )

    compiled_params = parameter_schema_spec_to_json_schema(spec.get("parameters", {}))
    compiled_output_schema = value_schema_spec_to_json_schema(
        raw_output["schema"], path="output.schema"
    )
    timeout_ms = validate_timeout_ms(spec.get("timeoutMs"))

    raw_execute = spec.get("execute")
    if not callable(raw_execute):
        raise ValueError(  # noqa: TRY004
            f'tool "{name}" must declare a callable execute handler'
        )

    def _accepts_run_ctx(fn: Callable[..., Any]) -> bool:
        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            return True
        pos = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
        ]
        return len(pos) >= 2 or any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
        )

    pass_run_ctx = _accepts_run_ctx(raw_execute)

    def _validated_execute(args: Any, run_ctx: Any = None) -> Any:
        violations = validate_json_schema_value(compiled_params, args, name="")
        if violations:
            raise ToolArgsError(violations)
        if pass_run_ctx and run_ctx is not None:
            return raw_execute(args, run_ctx)
        return raw_execute(args)

    output_def: dict[str, Any] = {
        "schema": compiled_output_schema,
        "render": raw_output["render"],
    }
    if callable(raw_output.get("presentationMeta")):
        output_def["presentationMeta"] = raw_output["presentationMeta"]

    definition: dict[str, Any] = {
        "name": name,
        "description": description,
        "parameters": compiled_params,
        "output": output_def,
        "execute": _validated_execute,
    }

    if callable(spec.get("isConcurrencySafe")):
        raw_conc = spec["isConcurrencySafe"]

        def _soft_conc(args: Any) -> bool:
            if not _soft_validate_args(compiled_params, args):
                return False
            return raw_conc(args) is True

        definition["isConcurrencySafe"] = _soft_conc

    if callable(spec.get("presentCall")):
        raw_pc = spec["presentCall"]

        def _soft_present_call(args: Any) -> Any:
            if not _soft_validate_args(compiled_params, args):
                return None
            return raw_pc(args)

        definition["presentCall"] = _soft_present_call

    if callable(spec.get("presentResult")):
        raw_pr = spec["presentResult"]

        def _soft_present_result(args: Any, result: Any) -> Any:
            if not _soft_validate_args(compiled_params, args):
                return None
            return raw_pr(args, result)

        definition["presentResult"] = _soft_present_result

    if callable(spec.get("projectContent")):
        definition["projectContent"] = spec["projectContent"]
    if callable(spec.get("finalizeContent")):
        definition["finalizeContent"] = spec["finalizeContent"]
    if timeout_ms is not None:
        definition["timeoutMs"] = timeout_ms
    if spec.get("deferLoading") is True:
        definition["deferLoading"] = True

    return definition


def define_tool(
    spec: Mapping[str, Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    execute: Callable[..., Any | Awaitable[Any]] | None = None,
    **extra: Any,
) -> Any:
    """Define either a validated ToolDefinition (mapping spec) or a legacy Tool."""
    if spec is not None:
        if not isinstance(spec, Mapping):
            raise TypeError("define_tool spec must be a mapping")
        return define_tool_definition(spec)
    if "output" in extra:
        return define_tool_definition(
            {
                "name": name,
                "description": description,
                "parameters": parameters or {},
                "execute": execute,
                **extra,
            }
        )
    if name is None or description is None:
        raise ValueError(
            "define_tool requires a spec mapping or name/description kwargs"
        )
    return legacy_define_tool(
        name=name,
        description=description,
        parameters=parameters,
        execute=cast(Any, execute),
    )


__all__ = [
    "Tool",
    "ToolArgsError",
    "define_tool",
    "define_tool_definition",
    "parameter_schema_spec_to_json_schema",
    "project_tool_schema",
    "validate_args",
    "validate_timeout_ms",
    "validate_value",
    "value_schema_spec_to_json_schema",
]
