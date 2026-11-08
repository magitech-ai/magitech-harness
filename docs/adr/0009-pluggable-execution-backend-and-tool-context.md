# 0009. Pluggable Execution Backend and Tool Context Isolation

Date: 2026-11-07
Status: accepted

## Context

Domain tools such as shell execution and file modification historically execute directly against the ambient host operating system. This host coupling creates severe security risks in untrusted workspaces, prevents running tools inside isolated containers or remote sandboxes, and complicates hermetic testing. Furthermore, passing runtime parameters (such as execution signals, working directories, or session identifiers) to tools often results in internal variables leaking into model-facing JSON Schemas.

## Decision

We establish the `ExecutionBackend` protocol and tool context isolation:
1. **`ExecutionBackend` Protocol**: Defines abstract primitives for command execution (`run(command, ...)`), filesystem inspection, and environment queries. The default implementation is `LocalShellService` on the host, while alternate backends (Docker containers, remote micro-VMs, or sandboxed environments) implement the identical protocol.
2. **Context Parameter Injection without Schema Leakage**: When registering tools, any parameter annotated as an execution context (`ToolContext` or `AnimaContext`) is automatically excluded from the model-facing JSON Schema. The tool dispatcher injects the active context instance at invocation time.
3. **Privilege Budgets Over Wish Lists**: Restricted execution backends enforce safety budgets at initialization time (e.g. rejecting forbidden modules and attributes before execution begins) rather than relying on permissive runtime attribute filters.

## Consequences

- Domain tools are decoupled from the host operating system and execute uniformly across local, containerized, and sandboxed environments.
- Model-facing tool schemas remain concise, clean, and free of leaked internal runtime parameters.
- Execution boundaries can be configured per profile without modifying tool implementations.
