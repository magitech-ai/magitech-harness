# ADR 0010: Deterministic 4-Tier Configuration Layer Resolution

## Status
Accepted

## Date
2026-11-08

## Context
In the `.agents` protocol, configuration, skills, agent profiles, and extensions reside across multiple filesystem tiers. Prior architectures suffered from three critical vulnerabilities:
1. **Divergent Resolution Paths**: Different subsystems (component loaders, prompt resolvers, tool loaders) implemented custom resolution heuristics, causing extensions to install successfully in one path but fail to load in another.
2. **Untrusted Ambient CWD Execution**: Automatically probing `Path.cwd() / ".agents"` when running in an arbitrary directory allowed untrusted repositories to execute arbitrary code or inject prompts on startup.
3. **Multi-Tenancy & Fleet Policy Gaps**: Inability to separate machine-level enterprise fleet baselines (`/etc/magitech`) from user-level developer preferences (`~/.agents`).

## Decision
1. **Centralized Layer Resolution Engine (`ctx.layers`)**:
   Implement `LayerResolutionService` mounted on `ctx.layers` as the single canonical resolver for all `.agents` protocol directory paths.
2. **Deterministic 4-Tier Hierarchy**:
   Resolution proceeds strictly from base to highest precedence:
   - **`global`**: `/etc/magitech` (or overridden via `$MAGITECH_GLOBAL_DIR`). System-level and fleet policies.
   - **`user`**: `~/.agents/`. Developer personal preferences, global skills, and private provider credentials.
   - **`agent`**: `~/.agents/agents/<name>`. Named subagent persona instructions and specialized profiles.
   - **`project`**: `<project>/.agents/`. Project-specific overrides and version-controlled repository configuration.
3. **Security Invariant: Project Layer Anchored, Never Ambient**:
   The `project` layer is included **only** when an explicit `project_path` is provided to the resolver. It **never** defaults to ambient `Path.cwd()`. Unspecified project contexts omit the project layer entirely, neutralizing ambient repository hijacking.
4. **Precedence API**:
   - `find_first`: Queries in highest-to-base order (`project -> agent -> user -> global`), returning the highest-precedence override.
   - `find_all`: Queries in base-to-highest order (`global -> user -> agent -> project`), enabling layered configuration stacking.

## Consequences
- **Positive**: Eliminates resolution divergence across all components, skills, steering docs, and profiles.
- **Positive**: Eliminates ambient directory hijacking by enforcing explicit project binding.
- **Positive**: Provides enterprise and container fleet customization via `$MAGITECH_GLOBAL_DIR`.
