"""Runtime configuration, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    kb_dir: Path = field(default_factory=lambda: Path(os.getenv("KB_DIR", "./knowledge_base")))

    embedding_backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "fastembed"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))

    # LLM: "auto" | "anthropic" | "openai" | "none"
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto").lower())
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # Agent behaviour
    max_iterations: int = field(default_factory=lambda: _int("MAX_ITERATIONS", 3))
    top_k: int = field(default_factory=lambda: _int("TOP_K", 4))
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 120))

    # Web crawler
    web_enabled: bool = field(default_factory=lambda: _bool("WEB_ENABLED", True))
    crawl_max_pages: int = field(default_factory=lambda: _int("CRAWL_MAX_PAGES", 3))
    allow_private_urls: bool = field(default_factory=lambda: _bool("ALLOW_PRIVATE_URLS", False))

    def resolved_provider(self) -> str:
        if self.llm_provider == "auto":
            if os.getenv("ANTHROPIC_API_KEY"):
                return "anthropic"
            if os.getenv("OPENAI_API_KEY"):
                return "openai"
            return "none"
        return self.llm_provider
