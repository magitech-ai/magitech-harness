"""
Tool definition structures and validation.
"""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable
from pydantic import BaseModel, Field


class Tool(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    handler: Any = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, args: dict[str, Any]) -> Any:
        if self.handler is None:
            raise NotImplementedError(f"Tool {self.name} has no handler")
        res = self.handler(args)
        if inspect.isawaitable(res):
            return await res
        return res

    def to_schema(self) -> dict[str, Any]:
        """Generate standard JSON function calling schema."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for p_name, p_spec in self.parameters.items():
            if isinstance(p_spec, dict):
                prop_type = p_spec.get("type", "string")
                prop_desc = p_spec.get("description", "")
                is_req = p_spec.get("required", False)
            else:
                prop_type = "string"
                prop_desc = str(p_spec)
                is_req = False

            properties[p_name] = {
                "type": prop_type,
                "description": prop_desc,
            }
            if is_req:
                required.append(p_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


def define_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    execute: Callable[[dict[str, Any]], Any | Awaitable[Any]],
) -> Tool:
    """Helper to define a model-callable tool."""
    return Tool(
        name=name,
        description=description,
        parameters=parameters,
        handler=execute,
    )
