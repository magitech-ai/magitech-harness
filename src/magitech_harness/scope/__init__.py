"""Scoped-context primitive for identity-tagged registrations and event routing."""

from magitech_harness.scope._core import (
    CreateScopeOptions,
    Scope,
    Scoped,
    ScopeKey,
    ScopeParentBinding,
    bind_scope_parent,
    carrier_key_of,
    create_scope,
    emit_scoped,
    is_scope_carrier,
    scope_chain_of,
    scope_of,
    scope_parent_of,
    scope_target,
    waterfall_scoped,
)
from magitech_harness.scope.store import (
    AnonymousEntries,
    NamedEntries,
    ScopedLayers,
    ScopeLayer,
)

bindScopeParent = bind_scope_parent
scopeParentOf = scope_parent_of
scopeChainOf = scope_chain_of
createScope = create_scope
scopeOf = scope_of
scopeTarget = scope_target
isScopeCarrier = is_scope_carrier
carrierKeyOf = carrier_key_of

__all__ = [
    "AnonymousEntries",
    "CreateScopeOptions",
    "NamedEntries",
    "Scope",
    "ScopeKey",
    "ScopeLayer",
    "ScopeParentBinding",
    "Scoped",
    "ScopedLayers",
    "bindScopeParent",
    "bind_scope_parent",
    "carrierKeyOf",
    "carrier_key_of",
    "createScope",
    "create_scope",
    "emit_scoped",
    "isScopeCarrier",
    "is_scope_carrier",
    "scopeChainOf",
    "scopeOf",
    "scopeParentOf",
    "scopeTarget",
    "scope_chain_of",
    "scope_of",
    "scope_parent_of",
    "scope_target",
    "waterfall_scoped",
]
