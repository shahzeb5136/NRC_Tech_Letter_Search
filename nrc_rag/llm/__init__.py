"""LLM provider adapters.

Every provider implements the same contract (:class:`nrc_rag.llm.base.LLMProvider`):
it receives retrieved excerpts and must return *claims with verbatim quotes tied
to chunk ids*. The provider never decides what is shown to the user - the
deterministic verifier does.
"""

from __future__ import annotations

import logging
from typing import Optional

from nrc_rag.config import Settings
from nrc_rag.llm.base import LLMProvider

log = logging.getLogger(__name__)


def get_provider(settings: Settings, name: Optional[str] = None) -> Optional[LLMProvider]:
    """Instantiate the configured provider (or ``None`` when no key is configured)."""
    name = name or settings.selected_provider()
    if name is None:
        return None
    if name == "anthropic":
        if not settings.anthropic_api_key:
            return None
        from nrc_rag.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings)
    if name == "openai":
        if not settings.openai_api_key:
            return None
        from nrc_rag.llm.openai_provider import OpenAICompatibleProvider

        return OpenAICompatibleProvider(settings)
    if name == "google":
        if not settings.google_api_key:
            return None
        from nrc_rag.llm.google_provider import GoogleProvider

        return GoogleProvider(settings)
    raise ValueError(f"unknown provider {name!r}")


__all__ = ["get_provider", "LLMProvider"]
