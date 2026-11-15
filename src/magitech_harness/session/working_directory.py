"""Session working-directory service (ctx.workingDirectory)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import Any, ClassVar
from weakref import WeakKeyDictionary

from anima import Context, EventMode, Service

from magitech_harness.agent.cancellation import (
    AbortController,
    AbortSignal,
    Deadline,
    combine_signals,
)
from magitech_harness.llm.message import create_user_message
from magitech_harness.session.projection_registry import ProjectionHandle
from magitech_harness.session.projections import WORKING_DIRECTORY_PROJECTION
from magitech_harness.session.store import Session


class WorkingDirectoryService(Service):
    """Single owner for each Session's effective directory and model-visible changes."""

    inject: ClassVar[list[str]] = ["fs", "sessionProjections", "systemPrompt"]

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, "workingDirectory")
        cfg = config or {}
        if "default_directory" in cfg or "defaultDirectory" in cfg:
            raw_dir = cfg.get("default_directory", cfg.get("defaultDirectory"))
            if not isinstance(raw_dir, str) or not raw_dir.strip():
                raise ValueError(
                    "working-directory: defaultDirectory must be a non-empty string"
                )
            directory = raw_dir
        else:
            directory = os.getcwd()
        if not os.path.isabs(directory):
            raise ValueError("working-directory: defaultDirectory must be absolute")
        self.default_directory = directory
        self.defaultDirectory = directory
        self._pending: WeakKeyDictionary[Session, asyncio.Task[Any]] = (
            WeakKeyDictionary()
        )
        self._operations: set[asyncio.Task[Any]] = set()
        self._lifecycle = AbortController()

        projections = ctx.get("sessionProjections", None, strict=False)
        if projections is None:
            raise RuntimeError("working-directory requires sessionProjections service")
        self._projection_handle: ProjectionHandle[str | None] = projections.register(
            WORKING_DIRECTORY_PROJECTION
        )

        system_prompt = ctx.get("systemPrompt", None, strict=False)
        if system_prompt is None:
            raise RuntimeError("working-directory requires systemPrompt service")
        ctx_disposer = system_prompt.context(
            {
                "name": "working-directory:current",
                "order": system_prompt.get_context_order("WORKING_DIRECTORY"),
                "required": True,
                "interpolate": False,
                "text": lambda context: (
                    ""
                    if context.get("agent") is None
                    else self._context_text(context["agent"].session)
                ),
            }
        )

        async def _on_assemble(
            assembly: dict[str, Any],
            context: Mapping[str, Any],
            next_fn: Callable[[], Awaitable[dict[str, Any]]],
        ) -> dict[str, Any]:
            agent = context.get("agent")
            if agent is None:
                return await next_fn()
            await self.ensure(agent, context.get("signal"))
            out = await next_fn()
            if self._lifecycle.signal.aborted:
                raise RuntimeError("working-directory service disposed")
            text = self._context_text(agent.session)
            contexts = out["contexts"]
            found = next(
                (
                    entry
                    for entry in contexts
                    if entry["name"] == "working-directory:current"
                ),
                None,
            )
            if found is None:
                contexts.insert(
                    0,
                    {
                        "name": "working-directory:current",
                        "text": text,
                        "interpolate": False,
                    },
                )
            else:
                found["text"] = text
                found["interpolate"] = False
            return out

        off_assemble = ctx.on(
            "system-prompt/assemble",
            _on_assemble,
            prepend=True,
            mode=EventMode.WATERFALL,
        )

        async def _teardown() -> None:
            self._lifecycle.abort(RuntimeError("working-directory service disposed"))
            if self._operations:
                await asyncio.gather(*self._operations, return_exceptions=True)
            off_assemble()
            ctx_disposer()
            self._projection_handle.dispose()

        ctx.effect(lambda: _teardown, label="working-directory lifecycle")

    def get(self, session: Session) -> str:
        """Read the committed working directory without filesystem I/O."""
        state = self._projection_handle.state_of(session)
        return state if state is not None else self.default_directory

    async def ensure(self, agent: Any, signal: AbortSignal | None = None) -> str:
        """Validate current directory and recover original project if missing."""

        async def _action(control: AbortSignal) -> str:
            current = self.get(agent.session)
            target = await self.ctx.fs.resolve(current, {"signal": control})
            info = await self.ctx.fs.stat(target, control)
            if control.aborted:
                raise self._abort_error(control)
            agent.ctx.fiber.assertActive()
            if info is not None and info["type"] == "directory":
                if self._projection_handle.state_of(agent.session) is None:
                    agent.session.append("working-directory/change", {"cwd": current})
                return current
            original = agent.session.header.get("cwd") or self.default_directory
            restored = await self._resolve_directory(original, original, control)
            self._commit(agent, restored, True, control)
            return restored

        return await self._serialize(agent, signal, _action)

    async def set(
        self, agent: Any, path: str, signal: AbortSignal | None = None
    ) -> str:
        """Change one Session's effective directory and queue a transition notice."""

        async def _action(control: AbortSignal) -> str:
            if len(path) == 0:
                raise ValueError("working-directory: cd must not be empty")
            directory = await self._resolve_directory(
                path, self.get(agent.session), control
            )
            self._commit(agent, directory, False, control)
            return directory

        return await self._serialize(agent, signal, _action)

    def _context_text(self, session: Session) -> str:
        return f"Current working directory: {json.dumps(self.get(session))}."

    @staticmethod
    def _abort_error(signal: AbortSignal) -> BaseException:
        if isinstance(signal.reason, BaseException):
            return signal.reason
        return RuntimeError("working-directory operation aborted")

    async def _resolve_directory(self, path: str, cwd: str, signal: AbortSignal) -> str:
        target = await self.ctx.fs.resolve(path, {"cwd": cwd, "signal": signal})
        info = await self.ctx.fs.stat(target, signal)
        if signal.aborted:
            raise self._abort_error(signal)
        if info is None or info["type"] != "directory":
            raise RuntimeError(
                f"working-directory: directory does not exist: {json.dumps(path)}"
            )
        return str(self.ctx.fs.process_path(target))

    def _commit(
        self,
        agent: Any,
        directory: str,
        recovered: bool,
        signal: AbortSignal,
    ) -> None:
        if signal.aborted:
            raise self._abort_error(signal)
        agent.ctx.fiber.assertActive()
        previous = self.get(agent.session)
        if previous == directory:
            return
        agent.session.append("working-directory/change", {"cwd": directory})
        text = (
            f"The working directory {json.dumps(previous)} is unavailable. "
            f"The working directory is now {json.dumps(directory)}."
            if recovered
            else f"The working directory changed from {json.dumps(previous)} to {json.dumps(directory)}."
        )
        notice = create_user_message(
            content=[{"type": "text", "text": text}],
            source={"kind": "working-directory"},
        )
        try:
            agent.inject(notice)
        except Exception as exc:  # noqa: BLE001 - committed change stands
            self.ctx.logger.warning(
                f"working-directory: committed {json.dumps(directory)}, "
                f"but could not queue its notice: {exc}"
            )

    async def _serialize(
        self,
        agent: Any,
        signal: AbortSignal | None,
        action: Callable[[AbortSignal], Coroutine[Any, Any, str]],
    ) -> str:
        session = agent.session
        fused = (
            Deadline(self._lifecycle.signal, lambda: None)
            if signal is None
            else combine_signals(self._lifecycle.signal, signal)
        )
        control = fused.signal
        previous = self._pending.get(session)

        async def _run() -> str:
            if previous is not None:
                # Wait sequentially across pending dir changes even if prior op aborted/failed.
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await previous
            if control.aborted:
                raise self._abort_error(control)
            agent.ctx.fiber.assertActive()
            return await action(control)

        task = asyncio.create_task(_run())
        self._operations.add(task)
        self._pending[session] = task
        try:
            return await task
        finally:
            self._operations.discard(task)
            fused.dispose()


__all__ = ["WorkingDirectoryService"]
