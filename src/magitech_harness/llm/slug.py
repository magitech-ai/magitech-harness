"""Model slug parsing utilities.

This module provides parsing functions for model identifiers in the format
``provider/model_name`` or bare ``model_name``.
"""

from __future__ import annotations


def parse_model_slug(slug: str) -> tuple[str, str]:
    """Parses a model slug string into its provider and model name parts.

    Model slugs can optionally specify a provider prefix separated by a slash,
    such as ``openrouter/meta-llama/llama-3-70b`` or ``openai/gpt-4o``. If no
    slash is present, the provider is returned as an empty string.

    Args:
        slug: The raw model slug string.

    Returns:
        A tuple of ``(provider_name, model_name)``. If no provider prefix is
        present, ``provider_name`` is the empty string.

    Examples:
        >>> parse_model_slug("openrouter/meta-llama/llama-3-70b")
        ('openrouter', 'meta-llama/llama-3-70b')
        >>> parse_model_slug("gpt-4o")
        ('', 'gpt-4o')
    """
    if not slug:
        return "", ""

    if "/" in slug:
        provider, model = slug.split("/", 1)
        return provider, model

    return "", slug
