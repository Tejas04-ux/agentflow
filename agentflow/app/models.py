"""Chat model factory (LangChain)."""
from __future__ import annotations

from .config import Settings


def get_chat_model(settings: Settings):
    """Return a LangChain chat model, or None when no provider/API key is configured."""
    provider = settings.resolved_provider()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.anthropic_model, max_tokens=1024, temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_model, temperature=0, max_tokens=1024)
    if provider == "none":
        return None
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (use auto, anthropic, openai or none)")
