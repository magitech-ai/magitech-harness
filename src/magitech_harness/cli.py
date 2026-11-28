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

from animas import Context

from magitech_harness import __version__, check_bundle_entries, create_harness
from magitech_harness.agent.cancellation import AbortError
from magitech_harness.agent.model_selection import install_model_selection
from magitech_harness.headless_json import HeadlessJsonProjection
from magitech_harness.llm.message import create_user_message
from magitech_harness.llm.slug import parse_model_slug
from magitech_harness.profiles.launcher import (
    dump_config,
    dump_config_schema,
    edit_components,
    resolve_profile_source,
)
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


async def _create_cli_agent(
    ctx: Context,
    model: str | None = None,
    session_id: str | None = None,
) -> tuple[Any, Session]:
    """Create an owned agent through the unified (Phase 3) driver.

    All CLI paths use this helper so ``repl`` and ``headless`` share
    one driver, one event namespace, and one plugin surface.
    """
    provider, model_name = _resolve_headless_route(ctx, model)
    options: dict[str, Any] = {
        "sessionId": session_id or f"cli-{uuid.uuid4()}",
        "agentOptions": {"provider": provider, "model": model_name},
    }
    handle = await ctx.agents.create(options)
    return handle["agent"], handle["agent"].session


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
    task: str | None,
    *,
    model: str | None = None,
    session_id: str | None = None,
    json_mode: bool = False,
) -> int:
    """Execute one headless turn via ``AgentRegistry`` and return its exit code."""
    try:
        if session_id is not None and session_id.strip() == "":
            raise ValueError("headless: session_id must not be blank")
        if task is None:
            task = await asyncio.to_thread(sys.stdin.read)
        assert task is not None  # stdin reader always returns a string
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
    projection: HeadlessJsonProjection | None = None
    reasoning_open = False

    def _close_reasoning() -> None:
        nonlocal reasoning_open
        if reasoning_open:
            print(file=sys.stderr)
            reasoning_open = False

    def _emit_reasoning(payload: dict[str, Any]) -> None:
        nonlocal reasoning_open
        message = payload.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        texts = (
            [
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "reasoning"
                and isinstance(block.get("text"), str)
                and block["text"] != ""
            ]
            if isinstance(content, list)
            else []
        )
        if not texts:
            return
        if not reasoning_open:
            print("mth: reasoning:", file=sys.stderr)
            reasoning_open = True
        print("".join(texts), file=sys.stderr, end="")

    def _on_agent_error(payload: dict[str, Any]) -> None:
        nonlocal failure
        if payload.get("agent") is agent:
            failure = (payload["error"], len(agent_session.events))

    def _on_session_event(subject: Session, event: Any) -> None:
        nonlocal failure
        if subject is not agent_session:
            return
        if (
            event.type == EventType.TURN_END
            and failure is not None
            and event.seq >= failure[1]
        ):
            failure = None
        if event.type == EventType.TURN_END:
            _close_reasoning()
        elif event.type == EventType.ASSISTANT_MESSAGE:
            _emit_reasoning(event.payload)
        if projection is not None:
            projection.on_event(event.type, event.payload, event.surface_op)

    stop_errors = ctx.on("agent/error", _on_agent_error)
    stop_terminals = ctx.on("session/event", _on_session_event)
    try:
        try:
            first_seq = len(agent.session.events)
            if json_mode:

                def _write_ndjson(line: str) -> None:
                    sys.stdout.write(line)

                projection = HeadlessJsonProjection(
                    _write_ndjson,
                    agent_session.id,
                    str(Path.cwd()),
                )
            agent.followup(
                create_user_message(
                    content=[{"type": "text", "text": task}],
                    source={"kind": "user"},
                )
            )
            await agent.when_idle()
            await _drain_and_flush(ctx)
            _close_reasoning()
            if failure is not None:
                if projection is not None:
                    projection.finish("", {"kind": "error", "message": str(failure[0])})
                return _report_headless_failure(failure[0], json_mode=json_mode)
            outcome = summarize_headless_outcome(agent.session, first_seq)
            exit_code = headless_exit_code(outcome)
            if projection is not None:
                projection.finish(outcome.text, outcome.reason)
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
            if projection is not None:
                projection.finish("", {"kind": "error", "message": str(exc)})
            return _report_headless_failure(exc, json_mode=json_mode)
    finally:
        stop_errors()
        stop_terminals()
        dispose_res = handle["dispose"]()
        if inspect.isawaitable(dispose_res):
            await dispose_res


class ReplDispatcher:
    """Dispatches interactive slash commands through ``ctx.commands``."""

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
        self._disposers: list[Any] = []
        self._install_commands()

    def _install_commands(self) -> None:
        from magitech_harness.commands.types import CommandDefinition, CommandResult

        self._disposers_by_name: dict[str, Any] = {}

        def register(
            name: str, description: str, handler: Any, *, input: Any = None
        ) -> None:
            # A newer dispatcher replaces the registration so its handlers
            # bind to the active REPL's session and model.
            if name in {
                definition.name for definition in self.ctx.commands.descriptors()
            }:
                self.ctx.commands.unregister(name)
            disposer = self.ctx.commands.register(
                CommandDefinition(
                    name=name, description=description, handler=handler, input=input
                )
            )
            self._disposers_by_name[name] = disposer
            self._disposers.append(disposer)

        def help_command(_invocation: Any) -> CommandResult:
            lines = [
                "Available slash commands:",
                "  /help             - Show this help message",
                "  /model [name]     - View or set active model slug",
                "  /session list     - List stored sessions",
                "  /session new      - Start a new session",
                "  /session show <id>- Inspect session events",
                "  /compact          - Trigger context compaction",
                "  /goal [arg]       - Show or manage the session goal",
                "  /feedback <text>  - Record session feedback",
                "  /clear            - Clear console screen",
                "  /exit, /quit      - Exit REPL",
            ]
            return CommandResult.success("\n" + "\n".join(lines))

        def model_command(invocation: Any) -> CommandResult:
            arg = invocation.raw_input.strip()
            if arg:
                self.model = arg
                return CommandResult.success(f"Active model set to: {self.model}")
            return CommandResult.success(f"Current model: {self.model or '(default)'}")

        def clear_command(_invocation: Any) -> CommandResult:
            self.out("\033[H\033[J", end="")
            return CommandResult.success()

        def session_command(invocation: Any) -> CommandResult:
            subparts = invocation.raw_input.strip().split(maxsplit=1)
            subcmd = subparts[0] if subparts else "list"
            subarg = subparts[1] if len(subparts) > 1 else ""
            if subcmd == "list":
                sessions = self.ctx.sessions.list_sessions()
                return CommandResult.success(
                    f"Sessions ({len(sessions)}): {', '.join(sessions) if sessions else 'none'}"
                )
            if subcmd == "new":
                self.session = self.ctx.sessions.create()
                return CommandResult.success(
                    f"Started new session: {self.session.session_id}"
                )
            if subcmd == "show":
                sess = self.ctx.sessions.read(subarg) if subarg else self.session
                if sess is None:
                    return CommandResult.error(f"Session not found: {subarg}")
                rows = [f"Session {sess.session_id}: {len(sess.events)} events"]
                rows.extend(
                    f"  [{event.type}] {event.payload}" for event in sess.events
                )
                return CommandResult.success("\n".join(rows))
            return CommandResult.error(f"Unknown session subcommand: {subcmd}")

        def goal_command(invocation: Any) -> CommandResult:
            from magitech_harness.goal.command import run_goal_command

            return CommandResult.success(
                run_goal_command(self.ctx, self.session, invocation.raw_input.strip())
            )

        async def compact_command(_invocation: Any) -> CommandResult:
            self.out("Invoking context compaction...")
            compact_res = await self.ctx.waterfall(
                "session/compact",
                {"session": self.session, "status": "requested"},
                default_handler=lambda payload: payload,
            )
            if compact_res.get("compacted"):
                return CommandResult.success(
                    f"Compacted: {compact_res.get('summary', 'Done')}"
                )
            return CommandResult.success(
                "Compaction completed or no compactable range found."
            )

        register("help", "Show this help message", help_command)
        register("model", "View or set active model slug", model_command)
        register("clear", "Clear console screen", clear_command)
        register("session", "List, create or inspect sessions", session_command)
        register("goal", "Show or manage the session goal", goal_command)
        register("compact", "Trigger context compaction", compact_command)
        self.ctx.effect(lambda: self._dispose_commands, "cli: dispose commands")

    def _dispose_commands(self) -> Any:
        def dispose() -> None:
            for disposer in self._disposers:
                disposer()
            self._disposers.clear()

        return dispose

    async def dispatch(self, cmd_line: str) -> bool:
        """Dispatch a slash command through the registry. Returns True to exit."""
        parts = cmd_line.strip().split(maxsplit=1)
        if parts and parts[0].lower() in ("/exit", "/quit"):
            self.out("Goodbye.")
            return True
        from types import SimpleNamespace

        agent = SimpleNamespace(id=self.session.session_id, session=self.session)
        result = await self.ctx.commands.execute(agent, cmd_line)
        if result is None:
            self.out(
                f"Unknown command: {parts[0] if parts else ''}. Type /help for available commands."
            )
        elif result.result.text:
            self.out(result.result.text)

        return False


async def run_repl(
    ctx: Context,
    session: Session | None = None,
    model: str | None = None,
    input_fn: Any = None,
    out_fn: Any = print,
) -> None:
    """Run the interactive REPL on the unified (Phase 3) agent driver.

    Streaming and tool activity are rendered from the driver's event bus
    (``agent/assistant-stream`` and ``session/event``) so the REPL and
    ``headless`` consume the same event namespace and plugin surface.
    """
    agent, current_session = await _create_cli_agent(ctx, model=model)
    dispatcher = ReplDispatcher(ctx, current_session, model=model, out=out_fn)

    out_fn("✨ Magitech Harness (mth) Interactive REPL")
    out_fn(f"Session: {current_session.id} | Model: {dispatcher.model or '(default)'}")
    out_fn("Type /help for slash commands, Ctrl+C to cancel turn, Ctrl+D to exit.\n")

    reader = input_fn or (lambda: input("mth> "))

    def _on_stream(payload: dict[str, Any]) -> None:
        frame = payload.get("frame") or {}
        if frame.get("type") != "chunk":
            return
        chunk = frame.get("chunk") or {}
        if isinstance(chunk, dict):
            text = chunk.get("text")
            if isinstance(
                text, str
            ):  # pragma: no branch - non-string chunk text is ignored
                out_fn(text, end="", flush=True)

    def _on_session_event(subject: Session, event: Any) -> None:
        if subject is not current_session:
            return
        if event.type == EventType.TOOL_CALL:
            payload = event.payload or {}
            out_fn(
                f"\n⚡ [Tool Start] {payload.get('name')}({payload.get('arguments')})"
            )
        elif event.type == EventType.TOOL_RESULT:
            payload = event.payload or {}
            status = "ERROR" if payload.get("is_error") else "OK"
            out_fn(f"\n✔️ [{status}] {payload.get('name')}")

    stop_stream = ctx.on("agent/assistant-stream", _on_stream)
    stop_events = ctx.on("session/event", _on_session_event)

    loop = asyncio.get_running_loop()
    sig_installed = False

    def _make_sigint_handler() -> Any:
        def _handler() -> None:
            agent.cancel({"kind": "user"})

        return _handler

    try:
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

            if not sig_installed:
                try:
                    import signal as py_signal

                    loop.add_signal_handler(py_signal.SIGINT, _make_sigint_handler())
                    sig_installed = True
                except (NotImplementedError, RuntimeError, AttributeError):
                    pass

            try:
                agent.followup(
                    create_user_message(
                        content=[{"type": "text", "text": line}],
                        source={"kind": "user"},
                    )
                )
                await agent.when_idle()
                out_fn("")
            except AbortError:
                out_fn("\n[Turn cancelled by user]")
            except Exception as e:  # noqa: BLE001
                out_fn(f"\n[Error during turn: {e}]")
    finally:
        stop_stream()
        stop_events()
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
        "--version",
        action="version",
        version=f"mth {__version__}",
        help="output the version number",
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
    parser.add_argument(
        "--patch",
        action="append",
        default=[],
        dest="patch",
        help="Extra patch-list overlay applied after the profile (repeatable)",
    )
    parser.add_argument(
        "--from-default-profile",
        dest="from_default_profile",
        help="Replace a named profile with a fresh copy of a shipped template, then boot it",
    )
    dumps = parser.add_mutually_exclusive_group()
    dumps.add_argument(
        "--dump-config",
        action="store_true",
        help="Print the composed profile tree and exit without booting",
    )
    dumps.add_argument(
        "--dump-default-config",
        action="store_true",
        help="Print the tree without the profile or --patch overlays and exit",
    )
    dumps.add_argument(
        "--dump-config-schema",
        action="store_true",
        help="Print JSON Schema for profile rows without mounting",
    )

    subparsers = parser.add_subparsers(dest="command")

    # repl subcommand
    subparsers.add_parser("repl", help="Start the interactive REPL shell")

    # headless subcommand
    headless_parser = subparsers.add_parser(
        "headless", help="Run one-shot prompt through the parity agent loop"
    )
    headless_parser.add_argument(
        "task",
        nargs="?",
        help="Task prompt to execute; omit or pass '-' to read stdin",
    )
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
    headless_parser.add_argument(
        "--script",
        dest="script",
        help=(
            "JSON file of scripted LLM stream chunks to drive the turn "
            "(overrides --model; offline tool-loop testing)"
        ),
    )

    # session subcommand
    session_parser = subparsers.add_parser(
        "session", help="Inspect and manage sessions"
    )
    session_sub = session_parser.add_subparsers(dest="session_command")
    session_sub.add_parser("list", help="List all sessions")
    show_p = session_sub.add_parser("show", help="Show session events")
    show_p.add_argument("session_id", help="Session ID to inspect")

    # plugin subcommand
    plugin_parser = subparsers.add_parser(
        "plugin", help="Manage a named profile's marketplace component list"
    )
    plugin_parser.add_argument(
        "--profile",
        dest="component_profile",
        required=True,
        help="Named profile whose component list to manage",
    )
    plugin_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Action (list/add/remove/show) and optional component name",
    )

    # component subcommand
    component_parser = subparsers.add_parser(
        "component", help="Inspect and manage components"
    )
    component_sub = component_parser.add_subparsers(dest="component_command")
    component_sub.add_parser("list", help="List installed/loaded components")
    component_sub.add_parser("discover", help="List components from installed packages")
    install_p = component_sub.add_parser(
        "install", help="Print the profile row for a component"
    )
    install_p.add_argument("name", help="Component name to install")

    return parser


async def main_async(argv: list[str] | None = None) -> int:
    """Asynchronous entry point for the mth CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    cmd = args.command
    dumps = args.dump_config or args.dump_default_config or args.dump_config_schema
    if dumps and cmd is not None:
        parser.error("config dumps take no subcommand")
    if args.from_default_profile and (args.profile is None or "/" in args.profile):
        parser.error("--from-default-profile needs a named profile")
    if args.dump_default_config and args.patch:
        parser.error("--dump-default-config takes no --patch")

    try:
        source = resolve_profile_source(args.profile, args.from_default_profile)
    except (OSError, ValueError) as exc:
        print(f"mth: {exc}", file=sys.stderr)
        return 1

    if dumps:
        if args.dump_config_schema:
            report = dump_config_schema()
        else:
            report = dump_config(
                None if args.dump_default_config else source,
                args.patch or None,
                default_only=args.dump_default_config,
            )
        print(json.dumps(report, indent=2))
        return 0

    if cmd == "plugin":
        parts = args.args or []
        if not parts or not parts[0]:
            parser.error("plugin needs an action (list/add/remove/show)")
        if len(parts) > 2:
            parser.error("plugin takes an action and at most one component name")
        try:
            components = edit_components(
                args.component_profile,
                parts[0],
                parts[1] if len(parts) == 2 else None,
            )
        except (OSError, ValueError) as exc:
            print(f"mth: {exc}", file=sys.stderr)
            return 1
        print("\n".join(components) if components else "(none)")
        return 0

    storage_path = Path(args.storage) if args.storage and args.storage.strip() else None
    ctx = create_harness(
        storage_dir=storage_path,
        profile=source,
        patches=args.patch or None,
    )
    await ctx.flush()
    check_bundle_entries(ctx)

    script_spec = None
    if getattr(args, "script", None):
        from magitech_harness.llm.scripted import ScriptedLlmAdapter, load_script

        try:
            script_spec = load_script(Path(args.script))
        except (OSError, ValueError) as exc:
            print(f"mth: invalid script: {exc}", file=sys.stderr)
            return 1
        # Title generation uses the legacy llm.call path, so keep a provider.
        if not ctx.llm.providers:
            from magitech_harness.llm.service import MockLLMProvider

            ctx.llm.register_provider("mock", MockLLMProvider())
        ctx.llm.register_adapter(
            [script_spec.provider], ScriptedLlmAdapter(script_spec)
        )
    elif not ctx.llm.providers and not ctx.llm.list_providers():
        from magitech_harness.llm.service import MockLlmAdapter, MockLLMProvider

        ctx.llm.register_provider("mock", MockLLMProvider())
        ctx.llm.register_adapter(["mock"], MockLlmAdapter())

    if cmd == "headless":
        effective_model = args.model
        if script_spec is not None:
            effective_model = f"{script_spec.provider}/{script_spec.model}"
        task = args.task
        if task == "-":
            task = None
        return await _run_headless(
            ctx,
            task,
            model=effective_model,
            session_id=args.session_id,
            json_mode=bool(args.json),
        )

    elif cmd == "session":
        if args.session_command == "show":
            if not args.session_id or not args.session_id.strip():
                print("mth: session id must not be blank", file=sys.stderr)
                return 1
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
        sub = args.component_command
        if sub == "discover":
            from magitech_harness.components.discovery import (
                discover_installed_components,
            )

            available = discover_installed_components()
            print(f"Available components ({len(available)}):")
            for record in available:
                version = record.get("version", "")
                description = record.get("description", "")
                suffix = f" ({version})" if version else ""
                print(f"  - {record['name']}{suffix} — {description}")
            return 0
        if sub == "install":
            from magitech_harness.components.discovery import (
                discover_installed_components,
            )

            available = discover_installed_components()
            match = next((r for r in available if r.get("name") == args.name), None)
            if match is None:
                print(f"mth: no such component: {args.name}", file=sys.stderr)
                return 1
            target = match.get("target")
            if not target:
                print(
                    f"mth: component '{args.name}' has no mount target",
                    file=sys.stderr,
                )
                return 1
            print("Add this row to your profile (or pass it via --profile):")
            print(f'  - "{target}"')
            return 0

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
                # entry.disabled is the evaluated boolean; the raw option may
                # be a JS expression node whose truthiness is always True.
                disabled = entry.disabled
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
