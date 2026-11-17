"""CLI entry point and interactive multi-turn REPL for magitech-harness.
Strictly adheres to: Everything is a Component.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anima import Context

from magitech_harness import check_bundle_entries, create_harness
from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.agent.model_selection import install_model_selection
from magitech_harness.llm.message import create_user_message
from magitech_harness.llm.slug import parse_model_slug
from magitech_harness.session.events import EventType
from magitech_harness.session.store import Session


@dataclass(frozen=True)
class HeadlessOutcome:
    """Aggregated final text and turn-end reason across one headless interval."""

    text: str
    reason: dict[str, Any] | None


def summarize_headless_outcome(session: Session, first_seq: int) -> HeadlessOutcome:
    """Summarize the last headless run interval; unclosed turns clear prior reasons."""
    started = False
    text = ""
    reason: dict[str, Any] | None = None
    for event in session.events[first_seq:]:
        if event.type == EventType.TURN_START:
            started = True
            reason = None
            continue
        if not started:
            continue
        if event.type == EventType.ASSISTANT_MESSAGE:
            msg = event.payload.get("message")
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, list):
                joined = "".join(
                    block["text"]
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                )
                if joined != "":
                    text = joined
        elif event.type == EventType.TURN_END:
            raw_reason = event.payload.get("reason")
            reason = dict(raw_reason) if isinstance(raw_reason, dict) else None
    return HeadlessOutcome(text=text, reason=reason)


def headless_exit_code(outcome: HeadlessOutcome) -> int:
    """Map one headless outcome to POSIX exit status: ``completed`` -> 0, else 1."""
    if outcome.reason is not None and outcome.reason.get("kind") == "completed":
        return 0
    return 1


async def _drain_and_flush(ctx: Context) -> None:
    """Drain background title work before flushing session persistence."""
    title_svc = ctx.get("sessionTitle", None, strict=False)
    if title_svc is not None:
        await title_svc.drain()
    await ctx.sessions.flush()


def _resolve_headless_route(ctx: Context, model_arg: str | None) -> tuple[str, str]:
    """Resolve ``(provider, model)`` for headless execution."""
    if model_arg and "/" in model_arg:
        provider, model = parse_model_slug(model_arg)
        return provider, model
    registered = ctx.llm.list_providers()
    provider = registered[0]["id"] if registered else "mock"
    model = model_arg or "mock"
    return provider, model


def _report_headless_failure(error: BaseException, *, json_mode: bool) -> int:
    """Emit structured headless failure output to stdout/stderr and return exit code 1."""
    message = str(error)
    if json_mode:
        print(json.dumps({"type": "error", "message": message}))
    print(f"mth: {message}", file=sys.stderr)
    return 1


async def _run_headless(
    ctx: Context,
    task: str,
    *,
    model: str | None = None,
    session_id: str | None = None,
    json_mode: bool = False,
) -> int:
    """Execute one headless turn via ``AgentRegistry`` and return its exit code."""
    try:
        if session_id is not None and session_id.strip() == "":
            raise ValueError("headless: session_id must not be blank")
        if task.strip() == "":
            raise ValueError(
                'a task is required, for example: mth headless "run the tests"'
            )

        provider_name, model_name = _resolve_headless_route(ctx, model)
        agent_options = {"provider": provider_name, "model": model_name}

        def _setup(agent_ctx: Context, created_agent: Any) -> None:
            if session_id is not None and model is not None:
                previous = created_agent.model.current()
                configured: dict[str, Any] = {
                    "provider": provider_name,
                    "model": model_name,
                }
                if (
                    previous is not None
                    and previous.get("provider") == provider_name
                    and previous.get("model") == model_name
                    and previous.get("reasoningEffort") is not None
                ):
                    configured["reasoningEffort"] = previous["reasoningEffort"]
                created_agent.model.select(configured)
            install_model_selection(agent_ctx)

        if session_id is not None:
            handle = await ctx.agents.resume(
                {
                    "resumeSessionId": session_id,
                    "agentOptions": agent_options,
                    "setup": _setup,
                }
            )
        else:
            fresh_id = f"headless-{uuid.uuid4()}"
            handle = await ctx.agents.create(
                {
                    "sessionId": fresh_id,
                    "agentOptions": agent_options,
                    "setup": _setup,
                }
            )
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        return _report_headless_failure(exc, json_mode=json_mode)

    agent = handle["agent"]
    agent_session = agent.session
    failure: tuple[BaseException, int] | None = None

    def _on_agent_error(payload: dict[str, Any]) -> None:
        nonlocal failure
        if payload.get("agent") is agent:
            failure = (payload["error"], len(agent_session.events))

    def _on_session_event(subject: Session, event: Any) -> None:
        nonlocal failure
        if (
            subject is agent_session
            and event.type == EventType.TURN_END
            and failure is not None
            and event.seq >= failure[1]
        ):
            failure = None

    stop_errors = ctx.on("agent/error", _on_agent_error)
    stop_terminals = ctx.on("session/event", _on_session_event)
    try:
        try:
            first_seq = len(agent.session.events)
            agent.followup(
                create_user_message(
                    content=[{"type": "text", "text": task}],
                    source={"kind": "user"},
                )
            )
            await agent.when_idle()
            await _drain_and_flush(ctx)
            if failure is not None:
                return _report_headless_failure(failure[0], json_mode=json_mode)
            outcome = summarize_headless_outcome(agent.session, first_seq)
            exit_code = headless_exit_code(outcome)
            if json_mode:
                print(
                    json.dumps(
                        {
                            "sessionId": agent.session.id,
                            "text": outcome.text,
                            "reason": outcome.reason,
                            "exitCode": exit_code,
                        }
                    )
                )
            else:
                print(outcome.text)
            if outcome.reason is not None and outcome.reason.get("kind") == "error":
                err = outcome.reason.get("error")
                if isinstance(err, dict):
                    code = err.get("code", "UNKNOWN")
                    msg = err.get("message", "")
                    print(f"mth: {code}: {msg}", file=sys.stderr)
            return exit_code
        except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
            return _report_headless_failure(exc, json_mode=json_mode)
    finally:
        stop_errors()
        stop_terminals()
        dispose_res = handle["dispose"]()
        if inspect.isawaitable(dispose_res):
            await dispose_res


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
                sess = self.ctx.sessions.read(subarg) if subarg else self.session
                if sess:
                    self.out(f"Session {sess.session_id}: {len(sess.events)} events")
                    for ev in sess.events:
                        self.out(f"  [{ev.type}] {ev.payload}")
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

    def _make_sigint_handler(sig: AbortSignal) -> Any:
        def _handler() -> None:
            sig.abort("SIGINT")

        return _handler

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
        loop = asyncio.get_running_loop()
        sig_installed = False

        try:
            import signal as py_signal

            loop.add_signal_handler(py_signal.SIGINT, _make_sigint_handler(signal))
            sig_installed = True
        except (NotImplementedError, RuntimeError, AttributeError):
            pass

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
                elif ev["type"] == "usage":
                    pass
                else:
                    out_fn("\n")
        except AbortError:
            out_fn("\n[Turn cancelled by user]")
        except Exception as e:  # noqa: BLE001
            out_fn(f"\n[Error during turn: {e}]")
        finally:
            if sig_installed:
                try:
                    import signal as py_signal

                    loop.remove_signal_handler(py_signal.SIGINT)
                except (RuntimeError, ValueError, OSError):
                    pass


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
        help="Optional YAML profile path to customize loaded components",
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

    # headless subcommand
    headless_parser = subparsers.add_parser(
        "headless", help="Run one-shot prompt through the parity agent loop"
    )
    headless_parser.add_argument("task", help="Task prompt to execute")
    headless_parser.add_argument(
        "--session-id",
        dest="session_id",
        help="Optional stored session ID to resume",
    )
    headless_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit final outcome as JSON",
    )

    # session subcommand
    session_parser = subparsers.add_parser(
        "session", help="Inspect and manage sessions"
    )
    session_sub = session_parser.add_subparsers(dest="session_command")
    session_sub.add_parser("list", help="List all sessions")
    show_p = session_sub.add_parser("show", help="Show session events")
    show_p.add_argument("session_id", help="Session ID to inspect")

    # component subcommand
    component_parser = subparsers.add_parser(
        "component", help="Inspect and manage components"
    )
    component_sub = component_parser.add_subparsers(dest="component_command")
    component_sub.add_parser("list", help="List installed/loaded components")

    return parser


async def main_async(argv: list[str] | None = None) -> int:
    """Asynchronous entry point for the mth CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    storage_path = Path(args.storage) if args.storage and args.storage.strip() else None
    ctx = create_harness(storage_dir=storage_path, profile=args.profile)
    await ctx.flush()
    check_bundle_entries(ctx)

    if not ctx.llm.providers and not ctx.llm.list_providers():
        from magitech_harness.llm.service import MockLLMProvider

        ctx.llm.register_provider("mock", MockLLMProvider())

    cmd = args.command

    if cmd == "run":
        session = ctx.sessions.create()
        print(f"\n🚀 Running prompt: '{args.prompt}'")
        res = await ctx.agent_loop.run_turn(session, args.prompt, model=args.model)
        await _drain_and_flush(ctx)
        print(f"\n🤖 Assistant: {res}")
        print(f"📝 Session Events Recorded: {len(session.events)}")
        location = ctx.sessions.location(session.session_id)
        if location is not None:
            print(f"💾 Persisted to: {location}")
        return 0

    if cmd == "headless":
        return await _run_headless(
            ctx,
            args.task,
            model=args.model,
            session_id=args.session_id,
            json_mode=bool(args.json),
        )

    elif cmd == "session":
        if args.session_command == "show":
            sess = ctx.sessions.read(args.session_id)
            if sess:
                print(f"Session '{sess.session_id}' ({len(sess.events)} events):")
                for ev in sess.events:
                    print(f"  [{ev.type}] {ev.payload}")
            else:
                print(f"Session not found: {args.session_id}")
        else:
            sessions = ctx.sessions.list_sessions()
            print(f"Stored sessions ({len(sessions)}):")
            for s in sessions:
                print(f"  - {s}")
        return 0

    elif cmd == "component":
        components = ctx.components.mounted_components
        print(f"Loaded components ({len(components)}):")
        for manifest, _fiber in components.values():
            print(f"  - {manifest.name} ({manifest.version})")
        loader = ctx.get("loader", None, strict=False)
        if loader is not None:
            entries = [e for e in loader.entries() if getattr(e, "options", None)]
            print(f"Bundle service rows ({len(entries)}):")
            for entry in entries:
                name = entry.options.get("name", "?")
                disabled = bool(entry.options.get("disabled"))
                state = "disabled" if disabled else "mounted"
                print(f"  - {entry.id} [bundle row] ({name}, {state})")
        return 0

    else:
        # Default or 'repl': run interactive REPL
        await run_repl(ctx, model=args.model)
        await _drain_and_flush(ctx)
        return 0


def main() -> None:
    """Synchronous launcher entry point."""
    try:
        code = asyncio.run(main_async(sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(0)
    if isinstance(code, int) and code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main()
