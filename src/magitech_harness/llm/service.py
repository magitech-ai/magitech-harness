"""LLMService (ctx.llm): adapter registry and streaming waterfall seam.

Provides the dual-surface LLM runtime for Magitech Harness:
1. Adapter registry, configurable-provider directory, draft model discovery,
   exact-model metadata & call-config resolution, one-shot prepared calls,
   durable file/image/tool-update projections, and terminal failure framing
   around the ``llm/stream`` waterfall.
2. Legacy provider compatibility layer (``LLMProvider``, ``MockLLMProvider``,
   ``NoProviderError``, ``register_provider``, ``call``) retained for existing
   callers until downstream loops migrate.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast, overload

from anima import Context, Service
from anima.events import Disposer

from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.llm.adapter_failure import (
    LlmError,
    assert_usable_api_key,
    assertUsableApiKey,
    normalize_llm_failure,
)
from magitech_harness.llm.call_config import (
    ConfigureCall,
    LlmCallConfig,
    LlmCallConfigAdapterDefaults,
    call_config_equals,
)
from magitech_harness.llm.content import (
    content_has_file,
    content_has_image,
    file_handle_text,
    project_files_to_text,
    project_images_for_text_model,
    project_tool_updates,
)
from magitech_harness.llm.error import RemoteError
from magitech_harness.llm.message import freeze_message
from magitech_harness.llm.retry_policy import resolve_retry_policy
from magitech_harness.llm.slug import parse_model_slug
from magitech_harness.llm.types import (
    FileAttachmentRef,
    GenerateOptions,
    ImageAttachmentRef,
    ImageRequestTarget,
    LlmConfigurableProvider,
    LlmDiscoveredModel,
    LlmFailure,
    LlmImageRequestPricing,
    LlmModelContext,
    LlmModelDiscoveryRequest,
    LlmModelInfo,
    LlmProviderInfo,
    LlmResolvedModelInfo,
    ModelChunk,
    ModelModality,
    ModelResponse,
    RequestMessage,
    StreamChunk,
    SystemPromptUpdate,
    ToolUpdate,
    UsageData,
)
from magitech_harness.llm.values import (
    deep_freeze,
    is_frozen,
    is_safe_integer,
)
from magitech_harness.session.events import ToolCallData


class NoProviderError(RuntimeError):
    """Raised when a legacy LLM call is attempted without an active provider."""


class LLMProvider(Protocol):
    """Legacy protocol implemented by Phase 2 model provider plugins."""

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generates a complete response from the underlying model."""
        ...  # pragma: no cover

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        signal: AbortSignal | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunk]:
        """Streams response chunks from the underlying model."""
        ...  # pragma: no cover


class MockLLMProvider:
    """Deterministic scriptable mock provider for offline tests and development."""

    def __init__(
        self,
        responses: list[ModelResponse] | None = None,
        *,
        default_model: str = "mock",
    ) -> None:
        """Initialize MockLLMProvider with an optional queue of responses."""
        self.responses: list[ModelResponse] = responses or []
        self.call_history: list[dict[str, Any]] = []
        self.default_model = default_model

    def queue_response(
        self,
        text: str = "",
        tool_calls: list[ToolCallData] | None = None,
        usage: UsageData | None = None,
    ) -> None:
        """Queue a mock response to be yielded by ``generate``."""
        self.responses.append(
            ModelResponse(
                text=text,
                tool_calls=tool_calls or [],
                finish_reason="tool_calls" if tool_calls else "stop",
                usage=usage or UsageData(),
            )
        )

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Yield the next queued response or a fallback default response."""
        self.call_history.append(
            {
                "messages": messages,
                "tools": tools,
                "system_prompt": system_prompt,
                "kwargs": kwargs,
            }
        )
        if self.responses:
            return self.responses.pop(0)
        return ModelResponse(text="Default mock response", tool_calls=[])

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        signal: AbortSignal | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunk]:
        """Yield mock response chunks as a stream."""
        resp = await self.generate(messages, tools, system_prompt, **kwargs)
        if signal and signal.is_aborted:
            raise AbortError("Operation was aborted.")
        if resp.text:
            yield ModelChunk(
                text=resp.text,
                tool_calls=[],
                finish_reason=None if resp.tool_calls else resp.finish_reason,
                usage=resp.usage,
            )
        if resp.tool_calls:
            yield ModelChunk(
                text="",
                tool_calls=resp.tool_calls,
                finish_reason=resp.finish_reason,
                usage=resp.usage,
            )
        if not resp.text and not resp.tool_calls:
            yield ModelChunk(
                text="",
                tool_calls=[],
                finish_reason=resp.finish_reason,
                usage=resp.usage,
            )


@dataclass(frozen=True)
class PreparedAdapterCall:
    """Adapter-owned model-resolution generation bound to its stream dispatch."""

    model: LlmResolvedModelInfo
    stream: Callable[[GenerateOptions], AsyncIterator[StreamChunk]]


class LlmAdapter:
    """Provider-wire adapter base class for the harness message and stream API."""

    def provider_info(self, provider: str) -> LlmProviderInfo:
        """Describe one provider route owned by this adapter."""
        return {"id": provider, "name": provider}

    def provider_retry_policy(self, provider: str) -> dict[str, Any] | None:
        """Return the provider-owned retry policy for *provider*, or None."""
        del provider
        return None

    def image_request_pricing(
        self, provider: str, model: str
    ) -> LlmImageRequestPricing | None:
        """Resolve provider-side request-image pricing for an exact route."""
        del provider, model
        return None

    def image_request_target(
        self,
        provider: str,
        model: str,
        image: ImageAttachmentRef,
    ) -> ImageRequestTarget:
        """Resolve image dimensions and byte target sent to the model."""
        del provider, model
        return {
            "width": image["width"],
            "height": image["height"],
            "maxBytes": image["bytes"],
        }

    async def list_models(self, provider: str) -> Sequence[LlmModelInfo]:
        """List models advertised for *provider* in adapter-preferred order."""
        del provider
        return []

    async def resolve_model(
        self,
        provider: str,
        model: str,
        signal: AbortSignal | None = None,
    ) -> LlmResolvedModelInfo:
        """Resolve metadata available for one exact route."""
        del signal
        return {"provider": provider, "id": model, "name": model}

    async def prepare_call(
        self,
        provider: str,
        model: str,
        signal: AbortSignal | None = None,
    ) -> PreparedAdapterCall:
        """Bind exact model metadata and stream dispatch to one generation."""
        resolved_model = await _call_adapter_async(
            self, "resolve_model", "resolveModel", provider, model, signal=signal
        )
        return PreparedAdapterCall(
            model=resolved_model,
            stream=lambda options: self.stream(options),
        )

    def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        """Stream raw model chunks for *options*."""
        del options
        raise NotImplementedError("LlmAdapter subclasses must implement stream()")


@dataclass(frozen=True)
class PreparedLlmCall:
    """One model call whose config and adapter registration resolved together."""

    config: LlmCallConfig
    retry_policy: dict[str, Any]
    adapter_defaults: LlmCallConfigAdapterDefaults
    _dispatch_stream: Callable[[GenerateOptions], AsyncIterator[StreamChunk]]
    context: LlmModelContext | None = None
    input_modalities: Sequence[ModelModality] | None = None
    system_prompt_update: SystemPromptUpdate | None = None
    tool_update: ToolUpdate | None = None

    @property
    def retryPolicy(self) -> dict[str, Any]:
        """CamelCase alias for ``retry_policy``."""
        return self.retry_policy

    @property
    def adapterDefaults(self) -> LlmCallConfigAdapterDefaults:
        """CamelCase alias for ``adapter_defaults``."""
        return self.adapter_defaults

    @property
    def inputModalities(self) -> Sequence[ModelModality] | None:
        """CamelCase alias for ``input_modalities``."""
        return self.input_modalities

    @property
    def systemPromptUpdate(self) -> SystemPromptUpdate | None:
        """CamelCase alias for ``system_prompt_update``."""
        return self.system_prompt_update

    @property
    def toolUpdate(self) -> ToolUpdate | None:
        """CamelCase alias for ``tool_update``."""
        return self.tool_update

    def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        """Dispatch this prepared call through its captured registration once."""
        return self._dispatch_stream(options)


class AdapterRegistrationHandle:
    """Disposable adapter registration handle supporting atomic route swaps."""

    def __init__(
        self,
        dispose: Callable[[], None],
        replace_fn: Callable[[Sequence[str]], None],
    ) -> None:
        self._dispose = dispose
        self._replace = replace_fn

    def __call__(self) -> None:
        """Release every route this registration currently holds."""
        self._dispose()

    def replace(self, providers: Sequence[str]) -> None:
        """Atomically replace this registration's routes."""
        self._replace(providers)


class DirectoryRegistrationHandle:
    """Disposable configurable-provider handle supporting atomic entry swaps."""

    def __init__(
        self,
        dispose: Callable[[], None],
        replace_fn: Callable[[Sequence[LlmConfigurableProvider]], None],
    ) -> None:
        self._dispose = dispose
        self._replace = replace_fn

    def __call__(self) -> None:
        """Withdraw every entry this registration currently holds."""
        self._dispose()

    def replace(self, entries: Sequence[LlmConfigurableProvider]) -> None:
        """Atomically replace this registration's configurable entries."""
        self._replace(entries)


@dataclass(frozen=True)
class _AdapterRegistration:
    adapter: Any
    provider: LlmProviderInfo
    retry_policy: dict[str, Any]


@dataclass(frozen=True)
class _PreparedDispatch:
    registration: _AdapterRegistration
    config: LlmCallConfig
    model_info: LlmResolvedModelInfo
    dispatch: Callable[[GenerateOptions], AsyncIterator[StreamChunk]]


_BASE_ADAPTER_METHODS: dict[str, Any] = {
    name: getattr(LlmAdapter, name)
    for name in (
        "provider_info",
        "provider_retry_policy",
        "image_request_pricing",
        "image_request_target",
        "list_models",
        "resolve_model",
        "prepare_call",
    )
}


def _resolve_adapter_callable(
    adapter: Any, snake_name: str, camel_name: str
) -> Callable[..., Any]:
    """Select an adapter override whether implemented in snake_case or camelCase."""
    base_fn = _BASE_ADAPTER_METHODS[snake_name]
    adapter_type = type(adapter)
    snake_fn = getattr(adapter, snake_name, None)
    if callable(snake_fn) and getattr(adapter_type, snake_name, None) is not base_fn:
        return cast(Callable[..., Any], snake_fn)
    camel_fn = getattr(adapter, camel_name, None)
    if callable(camel_fn):
        return cast(Callable[..., Any], camel_fn)
    if callable(snake_fn):
        return cast(Callable[..., Any], snake_fn)
    return cast(Callable[..., Any], base_fn.__get__(adapter, adapter_type))


def _invoke_with_optional_signal(
    fn: Callable[..., Any],
    *args: Any,
    signal: AbortSignal | None = None,
) -> Any:
    """Invoke *fn* forwarding *signal* when supported by the callable signature."""
    if signal is None:
        return fn(*args)
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return fn(*args, signal)
    params = list(sig.parameters.values())
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    positional_count = sum(
        1
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    )
    if has_varargs or positional_count > len(args) or "signal" in sig.parameters:
        return fn(*args, signal)
    return fn(*args)


def _call_adapter_sync(
    adapter: Any, snake_name: str, camel_name: str, *args: Any
) -> Any:
    """Invoke a synchronous adapter hook cleanly."""
    return _resolve_adapter_callable(adapter, snake_name, camel_name)(*args)


async def _call_adapter_async(
    adapter: Any,
    snake_name: str,
    camel_name: str,
    *args: Any,
    signal: AbortSignal | None = None,
) -> Any:
    """Invoke an async (or sync-returning) adapter hook cleanly."""
    fn = _resolve_adapter_callable(adapter, snake_name, camel_name)
    res = _invoke_with_optional_signal(fn, *args, signal=signal)
    if inspect.isawaitable(res):
        return await res
    return res


def _clone_plain(value: Any) -> Any:
    """Recursively copy values into fresh unfrozen dicts and lists."""
    if isinstance(value, Mapping):
        return {k: _clone_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_plain(item) for item in value]
    return copy.deepcopy(value)


def _signal_is_aborted(signal: Any) -> bool:
    """Return True when an abort signal has triggered."""
    if signal is None:
        return False
    return bool(
        getattr(signal, "is_aborted", False) or getattr(signal, "aborted", False)
    )


def _throw_if_aborted(signal: Any) -> None:
    """Raise when an abort signal has already fired."""
    if signal is None:
        return
    throw_fn = getattr(signal, "throw_if_aborted", None)
    if callable(throw_fn):
        throw_fn()
        return
    if _signal_is_aborted(signal):
        reason = getattr(signal, "reason", None)
        raise AbortError(str(reason) if reason else "Operation was aborted.")


def _adapter_failure_chunk(error: BaseException, signal: Any = None) -> StreamChunk:
    """Convert one adapter throw into the stream protocol's terminal finish chunk."""
    failure: LlmFailure = normalize_llm_failure(error)
    kind = (
        "aborted"
        if _signal_is_aborted(signal) or failure["code"] == "ABORTED"
        else "error"
    )
    return {
        "type": "finish",
        "reason": {"kind": kind, "failure": failure},
    }


def _detach_directory_entry(entry: LlmConfigurableProvider) -> LlmConfigurableProvider:
    """Copy a directory entry including its sequence fields."""
    copied: dict[str, Any] = {
        **entry,
        "settingsPath": list(entry["settingsPath"]),
    }
    if "baseURLEnv" in entry and entry["baseURLEnv"] is not None:
        copied["baseURLEnv"] = list(entry["baseURLEnv"])
    if "nativeAuth" in entry and entry["nativeAuth"] is not None:
        native = entry["nativeAuth"]
        copied["nativeAuth"] = {
            **native,
            "env": list(native["env"]),
        }
    return cast(LlmConfigurableProvider, copied)


def _sync_dispose(dispose: Disposer) -> Callable[[], None]:
    """Wrap an anima Disposer into a synchronous zero-argument handle."""

    def _close() -> None:
        _ = dispose()

    return _close


def _extract_prepared_adapter_call(raw: Any) -> PreparedAdapterCall:
    """Normalize either PreparedAdapterCall or mapping from adapter prepare_call."""
    if isinstance(raw, PreparedAdapterCall):
        return raw
    if isinstance(raw, Mapping):
        return PreparedAdapterCall(model=raw["model"], stream=raw["stream"])
    return PreparedAdapterCall(model=raw.model, stream=raw.stream)


async def _close_async_iterator(iterator: Any) -> None:
    """Await iterator cleanup when the underlying stream exposes aclose/return."""
    aclose = getattr(iterator, "aclose", None)
    if callable(aclose):
        res = aclose()
        if inspect.isawaitable(res):
            await res


class LLMService(Service):
    """Abstract ``llm`` service: adapter registry and streaming waterfall."""

    def __init__(self, ctx: Context, config: dict[str, Any] | None = None) -> None:
        """Initialize LLMService on *ctx*."""
        super().__init__(ctx, "llm")
        self._adapters: dict[str, _AdapterRegistration] = {}
        self._directory: dict[str, LlmConfigurableProvider] = {}
        self._discoveries: dict[
            str,
            Callable[
                ...,
                Awaitable[Sequence[LlmDiscoveredModel]] | Sequence[LlmDiscoveredModel],
            ],
        ] = {}
        self._listener_tasks: set[asyncio.Task[Any]] = set()
        self._providers: dict[str, LLMProvider] = {}
        self._active_provider_name: str | None = None

        provider = (config or {}).get("provider")
        if provider is not None:
            self.register_provider("default", provider)

    # ------------------------------------------------------------------
    # Legacy Phase 2 provider API (retained during incremental migration)
    # ------------------------------------------------------------------

    def register_provider(self, name: str, provider: LLMProvider) -> Disposer:
        """Register a named legacy model provider component."""
        self._providers[name] = provider
        if self._active_provider_name is None:
            self._active_provider_name = name

        def _unregister() -> None:
            if name in self._providers:
                del self._providers[name]
                if self._active_provider_name == name:
                    self._active_provider_name = (
                        next(iter(self._providers.keys())) if self._providers else None
                    )

        self.ctx.effect(lambda: _unregister)
        return _unregister

    def get_provider(self, name: str) -> LLMProvider | None:
        """Retrieve a registered legacy provider by name."""
        return self._providers.get(name)

    def set_active_provider(self, name: str) -> None:
        """Set the active legacy provider by name."""
        if name not in self._providers:
            raise KeyError(
                f"Provider '{name}' is not registered. "
                f"Available providers: {list(self._providers.keys())}"
            )
        self._active_provider_name = name

    @property
    def active_provider(self) -> LLMProvider | None:
        """Return the currently active legacy LLMProvider instance or None."""
        if self._active_provider_name is not None:
            return self._providers.get(self._active_provider_name)
        return None

    @property
    def providers(self) -> dict[str, LLMProvider]:
        """Return a mapping of all registered legacy providers."""
        return dict(self._providers)

    async def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Execute a legacy model call through the ``llm/call`` waterfall."""
        provider_name, _ = parse_model_slug(model or "")
        target_provider: LLMProvider | None = None

        if provider_name and provider_name in self._providers:
            target_provider = self._providers[provider_name]
        elif self.active_provider is not None:
            target_provider = self.active_provider

        if target_provider is None:
            raise NoProviderError(
                "No active LLM provider component registered. Please install or "
                "configure a provider component (e.g. 'mth component install "
                "openrouter') or specify one in your profile.yaml."
            )

        async def _call_provider(
            msgs: list[dict[str, Any]],
            tls: list[dict[str, Any]] | None,
            sys_p: str | None,
        ) -> ModelResponse:
            return await target_provider.generate(
                messages=msgs,
                tools=tls,
                system_prompt=sys_p,
                model=model,
                **kwargs,
            )

        response: ModelResponse = await self.ctx.waterfall(
            "llm/call",
            messages,
            tools,
            system_prompt,
            default_handler=_call_provider,
        )
        return response

    # ------------------------------------------------------------------
    # Parity LlmRuntime adapter registry, discovery, metadata & streaming
    # ------------------------------------------------------------------

    def _warn_adapters_listener_failure(self, error: Any) -> None:
        self.ctx.logger.warning("llm: an llm/adapters-updated listener failed")
        self.ctx.logger.warning(error)

    async def _await_listener_safely(self, awaitable: Awaitable[Any]) -> None:
        try:
            await awaitable
        except Exception as exc:  # noqa: BLE001
            # Contained topology observer rejections warn instead of failing emits.
            self._warn_adapters_listener_failure(exc)

    def _emit_adapters_updated(self) -> None:
        bus = self.ctx.events
        for listener in bus._filter_listeners(
            "llm/adapters-updated", this_arg=self.ctx
        ):
            try:
                returned = listener.callback()
            except Exception as exc:  # noqa: BLE001
                # Contained topology observer throws cannot starve later listeners.
                self._warn_adapters_listener_failure(exc)
                continue
            if inspect.isawaitable(returned):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    # Run outside an active event loop during synchronous teardown.
                    asyncio.run(self._await_listener_safely(returned))
                else:
                    task = loop.create_task(self._await_listener_safely(returned))
                    self._listener_tasks.add(task)
                    task.add_done_callback(self._listener_tasks.discard)

    def _registration(self, provider: str) -> _AdapterRegistration:
        reg = self._adapters.get(provider)
        if reg is None:
            raise LlmError(
                f'no adapter registered for provider "{provider}"',
                "NO_ADAPTER",
            )
        return reg

    def _prepare_routes(
        self,
        providers: Sequence[str],
        adapter: Any,
        owned: set[str],
    ) -> list[_AdapterRegistration]:
        unique: set[str] = set()
        registrations: list[_AdapterRegistration] = []
        for provider in providers:
            if not isinstance(provider, str) or len(provider) == 0:
                raise LlmError(
                    "adapter provider names must be non-empty",
                    "INVALID_ADAPTER",
                )
            if provider in unique or (
                provider in self._adapters and provider not in owned
            ):
                raise LlmError(
                    f'an adapter for provider "{provider}" is already registered',
                    "DUPLICATE_ADAPTER",
                )
            info = _call_adapter_sync(
                adapter, "provider_info", "providerInfo", provider
            )
            if (
                not isinstance(info, Mapping)
                or not isinstance(info.get("id"), str)
                or info["id"] != provider
                or not isinstance(info.get("name"), str)
                or len(info["name"]) == 0
            ):
                raise LlmError(
                    f'adapter metadata for provider "{provider}" must preserve its id '
                    "and have a non-empty name",
                    "INVALID_ADAPTER",
                )
            unique.add(provider)
            raw_policy = _call_adapter_sync(
                adapter, "provider_retry_policy", "providerRetryPolicy", provider
            )
            retry_policy = (
                raw_policy
                if raw_policy is not None
                else resolve_retry_policy(
                    None, f'llm: provider "{provider}" retryPolicy'
                )
            )
            registrations.append(
                _AdapterRegistration(
                    adapter=adapter,
                    provider={"id": info["id"], "name": info["name"]},
                    retry_policy=retry_policy,
                )
            )
        return registrations

    def _commit_routes(
        self,
        owned: set[str],
        registrations: Sequence[_AdapterRegistration],
    ) -> None:
        for provider in owned:
            self._adapters.pop(provider, None)
        owned.clear()
        for reg in registrations:
            pid = reg.provider["id"]
            self._adapters[pid] = reg
            owned.add(pid)
        self._emit_adapters_updated()

    def register_adapter(
        self, providers: Sequence[str], adapter: Any
    ) -> AdapterRegistrationHandle:
        """Register an adapter for *providers* (all-or-nothing, fiber-scoped)."""
        owned: set[str] = set()
        released = False

        def _setup() -> Callable[[], None]:
            if len(providers) == 0:
                raise LlmError(
                    "an adapter must register at least one provider",
                    "INVALID_ADAPTER",
                )
            self._commit_routes(owned, self._prepare_routes(providers, adapter, owned))

            def _teardown() -> None:
                nonlocal released
                released = True
                for provider in owned:
                    self._adapters.pop(provider, None)
                owned.clear()
                self._emit_adapters_updated()

            return _teardown

        dispose = self.ctx.effect(_setup, "llm.registerAdapter()")

        def _replace(next_providers: Sequence[str]) -> None:
            if released:
                raise LlmError(
                    "a disposed adapter registration cannot replace its routes",
                    "REGISTRATION_DISPOSED",
                )
            self._commit_routes(
                owned, self._prepare_routes(next_providers, adapter, owned)
            )

        return AdapterRegistrationHandle(_sync_dispose(dispose), _replace)

    def list_providers(self) -> list[LlmProviderInfo]:
        """Describe provider routes with a registered adapter in order."""
        return [
            {"id": reg.provider["id"], "name": reg.provider["name"]}
            for reg in self._adapters.values()
        ]

    def register_configurable_providers(
        self, entries: Sequence[LlmConfigurableProvider]
    ) -> DirectoryRegistrationHandle:
        """Declare configurable provider routes (all-or-nothing, fiber-scoped)."""
        held: list[LlmConfigurableProvider] = []
        disposed = False

        def _commit(candidates: Sequence[LlmConfigurableProvider]) -> None:
            nonlocal held
            detached: list[LlmConfigurableProvider] = []
            own = {entry["provider"] for entry in held}
            seen_candidates: set[str] = set()
            for entry in candidates:
                provider = entry.get("provider", "")
                display_name = entry.get("displayName", "")
                settings_ns = entry.get("settingsNs", "")
                settings_path = entry.get("settingsPath", ())
                if (
                    not isinstance(provider, str)
                    or len(provider) == 0
                    or not isinstance(display_name, str)
                    or len(display_name) == 0
                    or not isinstance(settings_ns, str)
                    or len(settings_ns) == 0
                ):
                    raise LlmError(
                        "configurable providers need a non-empty provider, "
                        "displayName, and settingsNs",
                        "INVALID_DIRECTORY",
                    )
                if any(
                    not isinstance(seg, str) or len(seg) == 0 for seg in settings_path
                ):
                    raise LlmError(
                        f'configurable provider "{provider}" has an empty '
                        "settingsPath segment",
                        "INVALID_DIRECTORY",
                    )
                if (
                    provider in self._directory and provider not in own
                ) or provider in seen_candidates:
                    raise LlmError(
                        f'configurable provider "{provider}" is already declared',
                        "DUPLICATE_DIRECTORY",
                    )
                seen_candidates.add(provider)
                detached.append(_detach_directory_entry(entry))

            for entry in held:
                self._directory.pop(entry["provider"], None)
            for entry in detached:
                self._directory[entry["provider"]] = entry
            held = detached
            self._emit_adapters_updated()

        def _setup() -> Callable[[], None]:
            if len(entries) == 0:
                raise LlmError(
                    "a configurable-provider registration must declare at least "
                    "one provider",
                    "INVALID_DIRECTORY",
                )
            _commit(entries)

            def _teardown() -> None:
                nonlocal disposed, held
                disposed = True
                for entry in held:
                    self._directory.pop(entry["provider"], None)
                held = []
                self._emit_adapters_updated()

            return _teardown

        dispose = self.ctx.effect(_setup, "llm.registerConfigurableProviders()")

        def _replace(next_entries: Sequence[LlmConfigurableProvider]) -> None:
            if disposed:
                raise LlmError(
                    "this configurable-provider registration was disposed",
                    "REGISTRATION_DISPOSED",
                )
            _commit(next_entries)

        return DirectoryRegistrationHandle(_sync_dispose(dispose), _replace)

    def list_configurable_providers(self) -> list[LlmConfigurableProvider]:
        """List every declared configurable provider, registered or dormant."""
        return [_detach_directory_entry(entry) for entry in self._directory.values()]

    def register_model_discovery(
        self,
        settings_ns: str,
        discover: Callable[
            ..., Awaitable[Sequence[LlmDiscoveredModel]] | Sequence[LlmDiscoveredModel]
        ],
    ) -> Callable[[], None]:
        """Offer draft-endpoint model discovery for *settings_ns*."""

        def _setup() -> Callable[[], None]:
            if not isinstance(settings_ns, str) or len(settings_ns) == 0:
                raise LlmError(
                    "model discovery needs a non-empty settings namespace",
                    "INVALID_DISCOVERY",
                )
            if settings_ns in self._discoveries:
                raise LlmError(
                    f'model discovery for "{settings_ns}" is already registered',
                    "DUPLICATE_DISCOVERY",
                )
            self._discoveries[settings_ns] = discover

            def _teardown() -> None:
                self._discoveries.pop(settings_ns, None)

            return _teardown

        dispose = self.ctx.effect(_setup, "llm.registerModelDiscovery()")
        return _sync_dispose(dispose)

    async def discover_models(
        self,
        settings_ns: str,
        request: LlmModelDiscoveryRequest,
        signal: AbortSignal | None = None,
    ) -> list[LlmDiscoveredModel]:
        """Interrogate one provider endpoint for the models it advertises."""
        discover = self._discoveries.get(settings_ns)
        if discover is None:
            raise LlmError(
                f'no model discovery is registered for "{settings_ns}"',
                "NO_DISCOVERY",
            )
        provider = request.get("provider") or ""
        base_url = request.get("baseURL") or ""
        if len(provider) == 0 and len(base_url) == 0:
            raise LlmError(
                "model discovery needs a provider route or a baseURL",
                "INVALID_DISCOVERY",
            )
        raw = _invoke_with_optional_signal(discover, request, signal=signal)
        discovered = await raw if inspect.isawaitable(raw) else raw
        seen: set[str] = set()
        models: list[LlmDiscoveredModel] = []
        for model in discovered:
            model_id = model.get("id")
            if not isinstance(model_id, str) or len(model_id) == 0 or model_id in seen:
                continue
            seen.add(model_id)
            item: dict[str, Any] = {"id": model_id}
            if "name" in model and model["name"] is not None:
                item["name"] = model["name"]
            if "contextWindow" in model and model["contextWindow"] is not None:
                item["contextWindow"] = model["contextWindow"]
            if "maxTokens" in model and model["maxTokens"] is not None:
                item["maxTokens"] = model["maxTokens"]
            if "inputModalities" in model and model["inputModalities"] is not None:
                item["inputModalities"] = list(model["inputModalities"])
            models.append(cast(LlmDiscoveredModel, item))
        return models

    async def remote_discover_models(
        self,
        settings_ns: str,
        request: LlmModelDiscoveryRequest,
        signal: AbortSignal | None = None,
    ) -> list[LlmDiscoveredModel]:
        """Remote discovery wrapper raising structured ``RemoteError`` on failure."""
        try:
            return await self.discover_models(settings_ns, request, signal)
        except Exception as err:
            # Remote RPC carriers wrap discovery rejections with protocol details.
            details: dict[str, Any] = {"settingsNs": settings_ns}
            if "baseURL" in request and request["baseURL"] is not None:
                details["baseURL"] = request["baseURL"]
            raise RemoteError(
                "llm/model-discovery-rejected",
                str(err),
                details,
                cause=err,
            ) from err

    def provider_retry_policy(self, provider: str) -> dict[str, Any]:
        """Resolve the retry policy captured when *provider* was registered."""
        return self._registration(provider).retry_policy

    def image_request_pricing(
        self, provider: str, model: str
    ) -> LlmImageRequestPricing | None:
        """Resolve provider-side request-image pricing, or None if unregistered."""
        reg = self._adapters.get(provider)
        if reg is None:
            return None
        return cast(
            LlmImageRequestPricing | None,
            _call_adapter_sync(
                reg.adapter,
                "image_request_pricing",
                "imageRequestPricing",
                provider,
                model,
            ),
        )

    def image_request_target(
        self,
        provider: str,
        model: str,
        image: ImageAttachmentRef,
    ) -> ImageRequestTarget:
        """Resolve image dimensions and byte target for an exact registered route."""
        reg = self._registration(provider)
        return cast(
            ImageRequestTarget,
            _call_adapter_sync(
                reg.adapter,
                "image_request_target",
                "imageRequestTarget",
                provider,
                model,
                image,
            ),
        )

    def _file_read_path(self, ref: FileAttachmentRef) -> str | None:
        host_path: str | None = None
        attachments = self.ctx.get("attachments", None, strict=False)
        if attachments is not None:
            resolver = getattr(attachments, "file_host_path", None) or getattr(
                attachments, "fileHostPath", None
            )
            if callable(resolver):
                try:
                    host_path = resolver(ref)
                except Exception:  # noqa: BLE001
                    # Malformed durable file refs degrade to no-path handle text.
                    return None
        if host_path is None:
            return None
        fs = self.ctx.get("fs", None, strict=False)
        if fs is None:
            return None
        mapper = getattr(fs, "process_path_from_host_path", None) or getattr(
            fs, "processPathFromHostPath", None
        )
        if not callable(mapper):
            return None
        return cast(str | None, mapper(host_path))

    def file_request_text(self, ref: FileAttachmentRef) -> str:
        """Resolve the exact text a durable file reference emits in requests."""
        return file_handle_text(ref, self._file_read_path(ref))

    @staticmethod
    def _detached_modalities(
        modalities: Sequence[ModelModality] | None,
    ) -> list[ModelModality] | None:
        return None if modalities is None else list(modalities)

    async def list_models(self, provider: str) -> list[LlmModelInfo]:
        """Discover models advertised by one registered provider."""
        adapter = self._registration(provider).adapter
        models = await _call_adapter_async(
            adapter, "list_models", "listModels", provider
        )
        seen: set[str] = set()
        result: list[LlmModelInfo] = []
        for model in models:
            if not isinstance(model, Mapping):
                raise LlmError(
                    f'adapter returned invalid or duplicate model metadata for provider "{provider}"',
                    "INVALID_CATALOG",
                )
            m_provider = model.get("provider")
            m_id = model.get("id")
            m_name = model.get("name")
            m_desc = model.get("description") if "description" in model else None
            if (
                not isinstance(m_provider, str)
                or m_provider != provider
                or not isinstance(m_id, str)
                or len(m_id) == 0
                or not isinstance(m_name, str)
                or len(m_name) == 0
                or (
                    "description" in model
                    and m_desc is not None
                    and not isinstance(m_desc, str)
                )
                or m_id in seen
            ):
                raise LlmError(
                    f'adapter returned invalid or duplicate model metadata for provider "{provider}"',
                    "INVALID_CATALOG",
                )
            seen.add(m_id)
            item: dict[str, Any] = {
                "provider": m_provider,
                "id": m_id,
                "name": m_name,
            }
            if m_desc is not None:
                item["description"] = m_desc
            modalities = self._detached_modalities(model.get("inputModalities"))
            if modalities is not None:
                item["inputModalities"] = modalities
            badge = model.get("badge")
            if badge is not None:
                badge_text = badge["text"]
                detached_text: Any = (
                    badge_text if isinstance(badge_text, str) else dict(badge_text)
                )
                item["badge"] = {"text": detached_text, "tone": badge["tone"]}
            result.append(cast(LlmModelInfo, item))
        return result

    def _normalize_model_info(
        self,
        registration: _AdapterRegistration,
        model: str,
        resolved: Any,
    ) -> LlmResolvedModelInfo:
        provider = registration.provider["id"]
        if (
            not isinstance(resolved, Mapping)
            or not isinstance(resolved.get("provider"), str)
            or resolved["provider"] != provider
            or not isinstance(resolved.get("id"), str)
            or resolved["id"] != model
            or not isinstance(resolved.get("name"), str)
            or len(resolved["name"]) == 0
            or (
                "description" in resolved
                and resolved["description"] is not None
                and not isinstance(resolved["description"], str)
            )
        ):
            raise LlmError(
                f'adapter returned invalid exact model metadata for provider "{provider}" '
                f'model "{model}"',
                "INVALID_MODEL_INFO",
            )
        context = resolved.get("context")
        if context is not None:
            cw = context.get("contextWindow") if isinstance(context, Mapping) else None
            if not is_safe_integer(cw) or cw <= 0:
                raise LlmError(
                    f'adapter returned invalid context metadata for provider "{provider}" '
                    f'model "{model}"',
                    "INVALID_MODEL_CONTEXT",
                )
        input_modalities = self._detached_modalities(resolved.get("inputModalities"))
        sys_update = resolved.get("systemPromptUpdate")
        if sys_update is not None and sys_update != "in-history":
            raise LlmError(
                f'adapter returned invalid system prompt update mode for provider "{provider}" '
                f'model "{model}"',
                "INVALID_MODEL_INFO",
            )
        tool_update = resolved.get("toolUpdate")
        if tool_update is not None and tool_update not in (
            "in-history",
            "addition-only",
        ):
            raise LlmError(
                f'adapter returned invalid tool update mode for provider "{provider}" '
                f'model "{model}"',
                "INVALID_MODEL_INFO",
            )
        default_max_tokens = resolved.get("defaultMaxTokens")
        if default_max_tokens is not None and (
            not is_safe_integer(default_max_tokens) or default_max_tokens <= 0
        ):
            raise LlmError(
                f'adapter returned invalid default maxTokens for provider "{provider}" '
                f'model "{model}"',
                "INVALID_MODEL_MAX_TOKENS",
            )
        info: dict[str, Any] = {
            "provider": provider,
            "id": model,
            "name": resolved["name"],
        }
        if resolved.get("description") is not None:
            info["description"] = resolved["description"]
        if input_modalities is not None:
            info["inputModalities"] = input_modalities
        if context is not None:
            info["context"] = {"contextWindow": context["contextWindow"]}
        if default_max_tokens is not None:
            info["defaultMaxTokens"] = default_max_tokens
        if sys_update is not None:
            info["systemPromptUpdate"] = sys_update
        if tool_update is not None:
            info["toolUpdate"] = tool_update

        reasoning = resolved.get("reasoning")
        if reasoning is None:
            return cast(LlmResolvedModelInfo, info)
        efforts_raw = (
            reasoning.get("efforts") if isinstance(reasoning, Mapping) else None
        )
        if (
            not isinstance(efforts_raw, Sequence)
            or isinstance(efforts_raw, (str, bytes))
            or len(efforts_raw) == 0
        ):
            raise LlmError(
                f'adapter returned invalid reasoning metadata for provider "{provider}" '
                f'model "{model}"',
                "INVALID_MODEL_REASONING",
            )
        seen: set[str] = set()
        efforts: list[dict[str, Any]] = []
        for effort in efforts_raw:
            if not isinstance(effort, Mapping):
                raise LlmError(
                    f"adapter returned invalid or duplicate reasoning effort metadata "
                    f'for provider "{provider}" model "{model}"',
                    "INVALID_MODEL_REASONING",
                )
            e_id = effort.get("id")
            e_name = effort.get("name")
            e_desc = effort.get("description") if "description" in effort else None
            if (
                not isinstance(e_id, str)
                or len(e_id) == 0
                or not isinstance(e_name, str)
                or len(e_name) == 0
                or (
                    "description" in effort
                    and e_desc is not None
                    and not isinstance(e_desc, str)
                )
                or e_id in seen
            ):
                raise LlmError(
                    f"adapter returned invalid or duplicate reasoning effort metadata "
                    f'for provider "{provider}" model "{model}"',
                    "INVALID_MODEL_REASONING",
                )
            seen.add(e_id)
            item: dict[str, Any] = {"id": e_id, "name": e_name}
            if e_desc is not None:
                item["description"] = e_desc
            efforts.append(item)

        default_effort = reasoning.get("defaultEffort")
        if default_effort is not None and default_effort not in seen:
            raise LlmError(
                f"adapter returned an unknown default reasoning effort for provider "
                f'"{provider}" model "{model}"',
                "INVALID_MODEL_REASONING",
            )
        reasoning_out: dict[str, Any] = {"efforts": efforts}
        if default_effort is not None:
            reasoning_out["defaultEffort"] = default_effort
        info["reasoning"] = reasoning_out
        return cast(LlmResolvedModelInfo, info)

    async def _resolve_model_info_for(
        self,
        registration: _AdapterRegistration,
        model: str,
        signal: AbortSignal | None = None,
    ) -> LlmResolvedModelInfo:
        resolved = await _call_adapter_async(
            registration.adapter,
            "resolve_model",
            "resolveModel",
            registration.provider["id"],
            model,
            signal=signal,
        )
        return self._normalize_model_info(registration, model, resolved)

    async def resolve_model_info(
        self,
        provider: str,
        model: str,
        signal: AbortSignal | None = None,
    ) -> LlmResolvedModelInfo:
        """Resolve and validate metadata for one exact provider/model route."""
        return await self._resolve_model_info_for(
            self._registration(provider), model, signal
        )

    def _resolve_call_with_info(
        self,
        config: Mapping[str, Any],
        info: LlmResolvedModelInfo,
    ) -> tuple[LlmCallConfig, LlmModelContext | None, LlmResolvedModelInfo]:
        provider = config["provider"]
        model = config["model"]
        default_max_tokens = info.get("defaultMaxTokens")
        defaulted: Mapping[str, Any] = (
            {**config, "maxTokens": default_max_tokens}
            if config.get("maxTokens") is None and default_max_tokens is not None
            else config
        )

        reasoning = info.get("reasoning")
        requested = defaulted.get("reasoningEffort")
        resolved_config: Mapping[str, Any] = defaulted
        if reasoning is None:
            if requested is not None:
                raise LlmError(
                    f'provider "{provider}" model "{model}" does not support reasoning '
                    f'effort "{requested}"',
                    "UNSUPPORTED_REASONING_EFFORT",
                )
        else:
            effective = (
                requested if requested is not None else reasoning.get("defaultEffort")
            )
            if effective is not None:
                supported = any(
                    effort["id"] == effective for effort in reasoning["efforts"]
                )
                if not supported:
                    raise LlmError(
                        f'provider "{provider}" model "{model}" does not support '
                        f'reasoning effort "{effective}"',
                        "UNSUPPORTED_REASONING_EFFORT",
                    )
                if requested != effective:
                    resolved_config = {**defaulted, "reasoningEffort": effective}

        return (
            cast(LlmCallConfig, resolved_config),
            info.get("context"),
            info,
        )

    async def resolve_call_config(
        self,
        config: LlmCallConfig,
        signal: AbortSignal | None = None,
    ) -> LlmCallConfig:
        """Validate a call config against its exact model capability and defaults."""
        reg = self._registration(config["provider"])
        info = await self._resolve_model_info_for(reg, config["model"], signal)
        resolved_config, _, _ = self._resolve_call_with_info(config, info)
        return resolved_config

    async def prepare_call(
        self,
        config: LlmCallConfig,
        signal: AbortSignal | None = None,
        configure: ConfigureCall | None = None,
    ) -> PreparedLlmCall:
        """Resolve a one-shot call handle bound to its captured adapter generation."""
        _throw_if_aborted(signal)
        provider = config["provider"]
        model = config["model"]
        controls = {
            key: value
            for key, value in config.items()
            if key not in ("provider", "model")
        }
        registration = self._registration(provider)
        raw_adapter_call = await _call_adapter_async(
            registration.adapter,
            "prepare_call",
            "prepareCall",
            provider,
            model,
            signal=signal,
        )
        adapter_call = _extract_prepared_adapter_call(raw_adapter_call)
        _throw_if_aborted(signal)
        model_info = self._normalize_model_info(registration, model, adapter_call.model)
        if configure is None:
            proposed: dict[str, Any] = dict(config)
        else:
            configured = configure(
                deep_freeze(_clone_plain(controls)),
                deep_freeze(_clone_plain(model_info)),
            )
            proposed = {**configured, "provider": provider, "model": model}
        _throw_if_aborted(signal)
        resolved_cfg_raw, ctx_meta, _ = self._resolve_call_with_info(
            proposed, model_info
        )
        resolved_config: LlmCallConfig = deep_freeze(_clone_plain(resolved_cfg_raw))
        frozen_context: LlmModelContext | None = (
            None if ctx_meta is None else deep_freeze(_clone_plain(ctx_meta))
        )
        defaults_dict: dict[str, Any] = {}
        if (
            proposed.get("reasoningEffort") is None
            and resolved_config.get("reasoningEffort") is not None
        ):
            defaults_dict["reasoningEffort"] = True
        if (
            proposed.get("maxTokens") is None
            and resolved_config.get("maxTokens") is not None
        ):
            defaults_dict["maxTokens"] = True
        adapter_defaults: LlmCallConfigAdapterDefaults = deep_freeze(
            cast(LlmCallConfigAdapterDefaults, defaults_dict)
        )
        raw_modalities = model_info.get("inputModalities")
        frozen_modalities: Sequence[ModelModality] | None = (
            None
            if raw_modalities is None
            else cast(Sequence[ModelModality], deep_freeze(list(raw_modalities)))
        )

        dispatched = False

        def _dispatch(options: GenerateOptions) -> AsyncIterator[StreamChunk]:
            nonlocal dispatched
            if dispatched:
                raise LlmError(
                    "a prepared LLM call can only be dispatched once",
                    "INVALID_PREPARED_CALL",
                )
            if not call_config_equals(options, resolved_config):
                raise LlmError(
                    "prepared LLM call config changed before adapter dispatch",
                    "INVALID_PREPARED_CALL",
                )
            dispatched = True
            return self._stream_with_registration(
                options,
                _PreparedDispatch(
                    registration=registration,
                    config=resolved_config,
                    model_info=model_info,
                    dispatch=adapter_call.stream,
                ),
            )

        return PreparedLlmCall(
            config=resolved_config,
            retry_policy=registration.retry_policy,
            adapter_defaults=adapter_defaults,
            _dispatch_stream=_dispatch,
            context=frozen_context,
            input_modalities=frozen_modalities,
            system_prompt_update=model_info.get("systemPromptUpdate"),
            tool_update=model_info.get("toolUpdate"),
        )

    def _for_adapter(self, options: Mapping[str, Any], adapter: Any) -> dict[str, Any]:
        """Strip replayState from assistant messages owned by another adapter."""
        orig_messages = options["messages"]
        new_messages: list[RequestMessage] = []
        changed = False
        for message in orig_messages:
            if message.get("role") != "assistant":
                new_messages.append(message)
                continue
            source = message.get("source")
            if not isinstance(source, Mapping) or source.get("replayState") is None:
                new_messages.append(message)
                continue
            owner_reg = self._adapters.get(source["provider"])
            if owner_reg is not None and owner_reg.adapter is adapter:
                new_messages.append(message)
                continue
            changed = True
            new_messages.append(
                freeze_message(
                    {
                        **message,
                        "source": {
                            "kind": "model",
                            "provider": source["provider"],
                            "model": source["model"],
                        },
                    }
                )
            )
        if not changed:
            return cast(dict[str, Any], options)
        filtered = {**options, "messages": new_messages}
        return deep_freeze(filtered) if is_frozen(options) else filtered

    async def _adapter_stream(
        self,
        options: Mapping[str, Any],
        prepared: _PreparedDispatch | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Final adapter boundary converting adapter failures into finish chunks."""
        signal = options.get("signal")
        iterator: AsyncIterator[StreamChunk]
        try:
            registration = (
                prepared.registration
                if prepared is not None
                else self._registration(options["provider"])
            )
            adapter = registration.adapter
            if prepared is None:
                raw_adapter_call = await _call_adapter_async(
                    adapter,
                    "prepare_call",
                    "prepareCall",
                    options["provider"],
                    options["model"],
                    signal=signal,
                )
                adapter_call = _extract_prepared_adapter_call(raw_adapter_call)
                model_info = self._normalize_model_info(
                    registration, options["model"], adapter_call.model
                )
                resolved_config, _, _ = self._resolve_call_with_info(
                    options, model_info
                )
                dispatch = adapter_call.stream
            else:
                model_info = prepared.model_info
                resolved_config = prepared.config
                dispatch = prepared.dispatch

            if prepared is not None and not call_config_equals(
                options, resolved_config
            ):
                raise LlmError(
                    "prepared LLM call config changed before adapter dispatch",
                    "INVALID_PREPARED_CALL",
                )

            if call_config_equals(options, resolved_config):
                resolved_options: dict[str, Any] = cast(dict[str, Any], options)
            else:
                merged = {**options, **resolved_config}
                resolved_options = deep_freeze(merged) if is_frozen(options) else merged

            projected_messages: Sequence[RequestMessage] = resolved_options["messages"]
            if any(content_has_file(msg["content"]) for msg in projected_messages):
                projected_messages = project_files_to_text(
                    projected_messages, self._file_read_path
                )
            input_modalities = model_info.get("inputModalities")
            if (
                input_modalities is not None
                and "image" not in input_modalities
                and any(content_has_image(msg["content"]) for msg in projected_messages)
            ):
                projected_messages = project_images_for_text_model(projected_messages)

            projected_tools = project_tool_updates(
                projected_messages,
                resolved_options.get("tools"),
                model_info.get("toolUpdate"),
                resolved_options.get("toolHistory"),
            )
            projected_messages = projected_tools["messages"]
            projected_options = resolved_options
            if projected_messages is not resolved_options[
                "messages"
            ] or projected_tools.get("tools") is not resolved_options.get("tools"):
                updated_opts: dict[str, Any] = {
                    **resolved_options,
                    "messages": list(projected_messages),
                }
                projected_tool_list = projected_tools.get("tools")
                if projected_tool_list is not None:
                    updated_opts["tools"] = list(projected_tool_list)
                projected_options = (
                    deep_freeze(updated_opts)
                    if is_frozen(resolved_options)
                    else updated_opts
                )

            stream_obj = dispatch(
                cast(GenerateOptions, self._for_adapter(projected_options, adapter))
            )
            if inspect.isawaitable(stream_obj):
                stream_obj = await stream_obj
            iterator = aiter(stream_obj)
        except Exception as exc:  # noqa: BLE001
            # Adapter setup, projection, and dispatch failures emit a terminal chunk.
            yield _adapter_failure_chunk(exc, signal)
            return

        completed = False
        try:
            while True:
                try:
                    chunk = await anext(iterator)
                except StopAsyncIteration:
                    completed = True
                    return
                except Exception as exc:  # noqa: BLE001
                    # Adapter stream iteration failures emit a terminal finish chunk.
                    completed = True
                    yield _adapter_failure_chunk(exc, signal)
                    return
                yield chunk
        finally:
            if not completed:
                await _close_async_iterator(iterator)

    async def _stream_with_registration(
        self,
        options: GenerateOptions,
        prepared: _PreparedDispatch | None = None,
    ) -> AsyncIterator[StreamChunk]:
        stream = await self.ctx.waterfall(
            "llm/stream",
            options,
            default_handler=lambda current_options: self._adapter_stream(
                current_options, prepared
            ),
        )
        iterator = aiter(stream)
        completed = False
        try:
            while True:
                try:
                    chunk = await anext(iterator)
                except StopAsyncIteration:
                    completed = True
                    return
                except BaseException:
                    completed = True
                    raise
                yield chunk
        finally:
            if not completed:
                await _close_async_iterator(iterator)

    async def _stream_legacy(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        signal: AbortSignal | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunk]:
        provider_name, _ = parse_model_slug(model or "")
        target_provider: LLMProvider | None = None

        if provider_name and provider_name in self._providers:
            target_provider = self._providers[provider_name]
        elif self.active_provider is not None:
            target_provider = self.active_provider

        if target_provider is None:
            raise NoProviderError(
                "No active LLM provider component registered. Please install or "
                "configure a provider component (e.g. 'mth component install "
                "openrouter') or specify one in your profile.yaml."
            )

        if signal:
            signal.throw_if_aborted()

        provider = target_provider
        options: dict[str, Any] = {
            "messages": messages,
            "tools": tools,
            "system_prompt": system_prompt,
            "model": model,
            "signal": signal,
            "session_id": session_id,
        }

        async def _stream_provider(
            stream_options: dict[str, Any],
        ) -> AsyncIterator[ModelChunk]:
            request_signal = stream_options["signal"]
            self.ctx.emit(
                "llm/stream-start",
                {
                    "messages": stream_options["messages"],
                    "model": stream_options["model"],
                },
            )

            async for chunk in provider.stream(
                messages=stream_options["messages"],
                tools=stream_options["tools"],
                system_prompt=stream_options["system_prompt"],
                model=stream_options["model"],
                signal=request_signal,
                **kwargs,
            ):
                if request_signal:
                    request_signal.throw_if_aborted()
                self.ctx.emit("llm/chunk", chunk)
                yield chunk

            self.ctx.emit("llm/stream-end", {"model": stream_options["model"]})

        stream = await self.ctx.waterfall(
            "llm/stream", options, default_handler=_stream_provider
        )
        async for chunk in stream:
            yield chunk

    @overload
    def stream(self, options: GenerateOptions, /) -> AsyncIterator[StreamChunk]: ...

    @overload
    def stream(self, *, options: GenerateOptions) -> AsyncIterator[StreamChunk]: ...

    @overload
    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        signal: AbortSignal | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunk]: ...

    def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Stream one model call through the ``llm/stream`` waterfall."""
        options_kw = kwargs.pop("options", None)
        first = args[0] if args else None
        parity_options = options_kw if options_kw is not None else first
        if isinstance(parity_options, Mapping) and len(args) <= 1 and not kwargs:
            return self._stream_with_registration(cast(GenerateOptions, parity_options))
        messages_kw = kwargs.pop("messages", None)
        legacy_messages: list[dict[str, Any]] = (
            messages_kw
            if messages_kw is not None
            else (first if isinstance(first, list) else [])
        )
        legacy_kwargs: dict[str, Any] = {
            name: (args[idx] if len(args) > idx else kwargs.pop(name, None))
            for idx, name in enumerate(
                ("tools", "system_prompt", "model", "signal", "session_id"), start=1
            )
        }
        return self._stream_legacy(legacy_messages, **legacy_kwargs, **kwargs)

    # CamelCase parity aliases matching LlmRuntime interface
    registerAdapter = register_adapter
    listProviders = list_providers
    registerConfigurableProviders = register_configurable_providers
    listConfigurableProviders = list_configurable_providers
    registerModelDiscovery = register_model_discovery
    discoverModels = discover_models
    remoteDiscoverModels = remote_discover_models
    providerRetryPolicy = provider_retry_policy
    imageRequestPricing = image_request_pricing
    imageRequestTarget = image_request_target
    fileRequestText = file_request_text
    listModels = list_models
    resolveModelInfo = resolve_model_info
    resolveCallConfig = resolve_call_config
    prepareCall = prepare_call


LlmRuntime = LLMService

__all__ = [
    "AdapterRegistrationHandle",
    "DirectoryRegistrationHandle",
    "LLMProvider",
    "LLMService",
    "LlmAdapter",
    "LlmError",
    "LlmRuntime",
    "MockLLMProvider",
    "NoProviderError",
    "PreparedAdapterCall",
    "PreparedLlmCall",
    "assertUsableApiKey",
    "assert_usable_api_key",
]
