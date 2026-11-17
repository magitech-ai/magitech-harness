"""Wire protocol implementations for pi-ai provider routes."""

from __future__ import annotations

from magitech_harness.llm.pi_ai.protocols.anthropic_messages import (
    stream_anthropic_messages,
)
from magitech_harness.llm.pi_ai.protocols.google_generative_ai import (
    stream_google_generative_ai,
)
from magitech_harness.llm.pi_ai.protocols.openai_completions import (
    stream_openai_completions,
)
from magitech_harness.llm.pi_ai.protocols.openai_responses import (
    stream_openai_responses,
)

STREAMERS = {
    "openai-completions": stream_openai_completions,
    "openai-responses": stream_openai_responses,
    "anthropic-messages": stream_anthropic_messages,
    "google-generative-ai": stream_google_generative_ai,
}

__all__ = [
    "STREAMERS",
    "stream_anthropic_messages",
    "stream_google_generative_ai",
    "stream_openai_completions",
    "stream_openai_responses",
]
