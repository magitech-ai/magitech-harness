# 0002. Zero Built-in Domain Tools and Providers in Core Runtime

Date: 2026-11-07
Status: accepted

## Context

Many agent harnesses bundle built-in domain tools (e.g. filesystem manipulators, bash execution) and concrete LLM client SDKs directly into the core engine. This introduces unwanted dependencies, complicates hermetic testing, causes tight coupling to specific APIs, and prevents replacing or customizing foundational capabilities.

## Decision

We enforce the principle "Everything is a Plugin":
1. `magitech-harness` contains zero built-in domain tools and zero built-in model providers.
2. Services such as `LocalShellService` (`ctx.shell`) act strictly as internal execution infrastructure for plugins, and never register themselves as tools onto `ctx.tools`.
3. All domain tools (`fs-tool`, `bash-tool`), guardrails (`truncation-guard`, `compaction-guard`), and model providers (`openrouter-provider`, `pollinations-provider`) reside exclusively in the `mth-plugins` marketplace repository.
4. The core harness defines only the service protocols (`LLMProvider`, `ToolDefinition`, `ToolService`, `LLMService`).

## Consequences

- The core runtime engine is lightweight, decoupled, and has minimal external dependencies.
- Tools and model providers can be versioned, tested, and upgraded independently in the marketplace without modifying the engine.
- Out of the box, the engine requires at least one model provider plugin to execute turns.
