# 0007. Two-Layer Invariant Prompt Scaffolding

Date: 2026-11-07
Status: accepted

## Context

In complex agent environments, prompt templates often conflate static behavioral personas with dynamic runtime facts (such as active tools, host environment metadata, shell syntax rules, and project context). Hardcoding dynamic tool definitions or environment parameters into static markdown files leads to prompt drift, tool hallucinations when capabilities change, and friction during autonomous agent adaptation.

## Decision

We establish the **Two-Layer Invariant Scaffolding Pattern** anchored by `SystemPromptService` (`ctx.system_prompt`):
1. **Layer 1 (Persona & Static Guidelines)**: Defined in plain markdown files (e.g. `SYSTEM.md`, `GUIDELINES.md`) or caller configuration. Layer 1 focuses strictly on domain persona, conversational style, and role invariants, remaining completely free of hardcoded tool catalogs or runtime environment details.
2. **Layer 2 (Dynamic Invariant Scaffolding)**: Rendered dynamically at prompt compile time by `ctx.system_prompt.build()`:
   - `Active Tools`: Rendered automatically from the currently registered tools in `ctx.tools.list_tools()`.
   - `Environment`: Platform OS, machine architecture, working directory, and current UTC timestamp.
   - `Project Context`: Dynamically discovered and injected from project and workspace `AGENTS.md` steering files.
   - `Skills & Guidelines`: Summaries of discoverable skills and operational rules.
3. Components and services contribute sections to Layer 2 dynamically using `ctx.system_prompt.add_section(title, content) -> Disposer`.

## Consequences

- Persona files remain clean, portable, and decoupled from runtime toolsets or host platforms.
- Tool registrations, skill discoveries, and environment changes automatically reflect in prompt compilation with zero manual prompt synchronization.
- System prompt assembly remains modular and observable.
