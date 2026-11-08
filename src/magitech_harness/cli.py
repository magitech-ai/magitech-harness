"""
CLI entry point for magitech-harness.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from magitech_harness import create_harness, define_tool


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Magitech Harness: Composable Agent Runtime by Magitech"
    )
    parser.add_argument(
        "prompt", nargs="?", help="Prompt to run through the agent loop"
    )
    parser.add_argument(
        "--storage",
        default=".sessions",
        help="Directory to persist JSONL session logs (default: .sessions)",
    )
    parser.add_argument(
        "--profile",
        help="Optional YAML profile path to customize loaded plugins",
    )

    args = parser.parse_args()

    ctx = create_harness(storage_dir=args.storage)
    await ctx.flush()

    if args.profile:
        from anima.loader import load_profile

        load_profile(ctx, args.profile)
        await ctx.flush()
    if not ctx.llm.providers:
        from magitech_harness.llm.service import MockLLMProvider

        ctx.llm.register_provider("mock", MockLLMProvider())

    # Register an example custom tool
    def add_numbers(tool_args: dict) -> int:
        return int(tool_args.get("a", 0)) + int(tool_args.get("b", 0))

    ctx.tools.register(
        define_tool(
            name="add",
            description="Add two numbers together.",
            parameters={
                "a": {
                    "type": "integer",
                    "description": "First number",
                    "required": True,
                },
                "b": {
                    "type": "integer",
                    "description": "Second number",
                    "required": True,
                },
            },
            execute=add_numbers,
        )
    )

    prompt = args.prompt or "What is 2 + 2?"
    print(f"\n🚀 Running Harness Turn with prompt: '{prompt}'")

    session = ctx.sessions.create()
    result = await ctx.agent_loop.run_turn(session, prompt)

    print(f"\n🤖 Assistant: {result}")
    print(f"📝 Session Events Recorded: {len(session.events)}")
    if session.storage_path:
        print(f"💾 Persisted to: {session.storage_path}")


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
