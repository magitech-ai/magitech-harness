"""CLI entry point and interactive multi-turn REPL for magitech-harness.
Strictly adheres to: Everything is a Plugin.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from anima import Context
from anima.loader import load_profile

from magitech_harness import create_harness
from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.session.store import Session


class ReplDispatcher:
    """Dispatches interactive slash commands in the REPL."""

    def __init__(
        self,
        ctx: Context,
        session: Session,
        model: str | None = None,
        out: Any = print,
    ) -> None:
        self.ctx = ctx
        self.session = session
        self.model = model
        self.out = out

    async def dispatch(self, cmd_line: str) -> bool:
        """Dispatches a slash command. Returns True if REPL should exit."""
        parts = cmd_line.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            self.out("Goodbye.")
            return True

        if cmd == "/help":
            self.out(
                "\nAvailable slash commands:\n"
                "  /help             - Show this help message\n"
                "  /model [name]     - View or set active model slug\n"
                "  /session list     - List stored sessions\n"
                "  /session new      - Start a new session\n"
                "  /session show <id>- Inspect session events\n"
                "  /compact          - Trigger context compaction\n"
                "  /clear            - Clear console screen\n"
                "  /exit, /quit      - Exit REPL\n"
            )
        elif cmd == "/model":
            if arg:
                self.model = arg
                self.out(f"Active model set to: {self.model}")
            else:
                self.out(f"Current model: {self.model or '(default)'}")
        elif cmd == "/clear":
            self.out("\033[H\033[J", end="")
        elif cmd == "/session":
            subparts = arg.split(maxsplit=1)
            subcmd = subparts[0] if subparts else "list"
            subarg = subparts[1] if len(subparts) > 1 else ""
            if subcmd == "list":
                sessions = self.ctx.sessions.list_sessions()
                self.out(
                    f"Sessions ({len(sessions)}): {', '.join(sessions) if sessions else 'none'}"
                )
            elif subcmd == "new":
                self.session = self.ctx.sessions.create()
                self.out(f"Started new session: {self.session.session_id}")
            elif subcmd == "show":
                sess = self.ctx.sessions.load(subarg) if subarg else self.session
                if sess:
                    self.out(f"Session {sess.session_id}: {len(sess.events)} events")
                    for ev in sess.events:
                        self.out(f"  [{ev.type.value}] {ev.payload}")
                else:
                    self.out(f"Session not found: {subarg}")
            else:
                self.out(f"Unknown session subcommand: {subcmd}")
        elif cmd == "/compact":
            self.out("Invoking context compaction...")
            compact_res = await self.ctx.waterfall(
                "session/compact",
                {"session": self.session, "status": "requested"},
                default_handler=lambda p: p,
            )
            if compact_res.get("compacted"):
                self.out(f"Compacted: {compact_res.get('summary', 'Done')}")
            else:
                self.out("Compaction completed or no compactable range found.")
        else:
            self.out(f"Unknown command: {cmd}. Type /help for available commands.")

        return False


async def run_repl(
    ctx: Context,
    session: Session | None = None,
    model: str | None = None,
    input_fn: Any = None,
    out_fn: Any = print,
) -> None:
    """Runs the interactive multi-turn REPL with streaming and cooperative cancellation."""
    current_session = session or ctx.sessions.create()
    dispatcher = ReplDispatcher(ctx, current_session, model=model, out=out_fn)

    out_fn("✨ Magitech Harness (mth) Interactive REPL")
    out_fn(
        f"Session: {dispatcher.session.session_id} | Model: {dispatcher.model or '(default)'}"
    )
    out_fn("Type /help for slash commands, Ctrl+C to cancel turn, Ctrl+D to exit.\n")

    reader = input_fn or (lambda: input("mth> "))

    while True:
        try:
            line = await asyncio.to_thread(reader)
        except (EOFError, KeyboardInterrupt):
            out_fn("\nGoodbye.")
            break

        line = line.strip()
        if not line:
            continue

        if line.startswith("/"):
            should_exit = await dispatcher.dispatch(line)
            if should_exit:
                break
            continue

        signal = AbortSignal()
        try:
            async for ev in ctx.agent_loop.stream_turn(
                dispatcher.session, line, model=dispatcher.model, signal=signal
            ):
                if ev["type"] == "chunk":
                    out_fn(ev["text"], end="", flush=True)
                elif ev["type"] == "tool_start":
                    out_fn(f"\n⚡ [Tool Start] {ev['name']}({ev['arguments']})")
                elif ev["type"] == "tool_end":
                    status = "ERROR" if ev.get("is_error") else "OK"
                    out_fn(f"\n✔️ [{status}] {ev['name']} -> {ev['result']}")
                else:
                    out_fn("\n")
        except AbortError:
            out_fn("\n[Turn cancelled by user]")
        except Exception as e:  # noqa: BLE001
            out_fn(f"\n[Error during turn: {e}]")


def build_parser() -> argparse.ArgumentParser:
    """Constructs the CLI argument parser for mth."""
    parser = argparse.ArgumentParser(
        prog="mth",
        description="Magitech Harness: Composable Agent Runtime by Magitech",
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
    parser.add_argument(
        "--model",
        help="Model slug to use (e.g. 'openrouter/anthropic/claude-3.5-sonnet')",
    )

    subparsers = parser.add_subparsers(dest="command")

    # repl subcommand
    subparsers.add_parser("repl", help="Start the interactive REPL shell")

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run a one-shot prompt")
    run_parser.add_argument("prompt", help="Prompt to execute")

    # session subcommand
    session_parser = subparsers.add_parser(
        "session", help="Inspect and manage sessions"
    )
    session_sub = session_parser.add_subparsers(dest="session_command")
    session_sub.add_parser("list", help="List all sessions")
    show_p = session_sub.add_parser("show", help="Show session events")
    show_p.add_argument("session_id", help="Session ID to inspect")

    # plugin subcommand
    plugin_parser = subparsers.add_parser("plugin", help="Inspect and manage plugins")
    plugin_sub = plugin_parser.add_subparsers(dest="plugin_command")
    plugin_sub.add_parser("list", help="List installed/loaded plugins")

    return parser


async def main_async(argv: list[str] | None = None) -> None:
    """Asynchronous entry point for the mth CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    storage_path = Path(args.storage) if args.storage and args.storage.strip() else None
    ctx = create_harness(storage_dir=storage_path)
    await ctx.flush()

    if args.profile:
        load_profile(ctx, args.profile)
        await ctx.flush()

    if not ctx.llm.providers:
        from magitech_harness.llm.service import MockLLMProvider

        ctx.llm.register_provider("mock", MockLLMProvider())

    cmd = args.command

    if cmd == "run":
        session = ctx.sessions.create()
        print(f"\n🚀 Running prompt: '{args.prompt}'")
        res = await ctx.agent_loop.run_turn(session, args.prompt, model=args.model)
        print(f"\n🤖 Assistant: {res}")
        print(f"📝 Session Events Recorded: {len(session.events)}")
        if session.storage_path:
            print(f"💾 Persisted to: {session.storage_path}")

    elif cmd == "session":
        if args.session_command == "show":
            sess = ctx.sessions.load(args.session_id)
            if sess:
                print(f"Session '{sess.session_id}' ({len(sess.events)} events):")
                for ev in sess.events:
                    print(f"  [{ev.type.value}] {ev.payload}")
            else:
                print(f"Session not found: {args.session_id}")
        else:
            sessions = ctx.sessions.list_sessions()
            print(f"Stored sessions ({len(sessions)}):")
            for s in sessions:
                print(f"  - {s}")

    elif cmd == "plugin":
        plugins = ctx.plugins.mounted_plugins
        print(f"Loaded plugins ({len(plugins)}):")
        for manifest, _fiber in plugins.values():
            print(f"  - {manifest.name} ({manifest.version})")

    else:
        # Default or 'repl': run interactive REPL
        await run_repl(ctx, model=args.model)


def main() -> None:
    """Synchronous launcher entry point."""
    try:
        asyncio.run(main_async(sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
