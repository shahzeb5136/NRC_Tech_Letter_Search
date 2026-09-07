"""Application settings.

All configuration is read from environment variables or a ``.env`` file in the
project root. Secrets (API keys) must never be committed - see ``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_VERSION = "1.0.0"
# Bump when extraction / chunking logic changes so that existing indexes are rebuilt.
PIPELINE_VERSION = "2026.09.1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ paths
    data_dir: Path = Field(default=PROJECT_ROOT / "Data", validation_alias=AliasChoices("NRC_DATA_DIR"))
    index_dir: Path = Field(default=PROJECT_ROOT / "index", validation_alias=AliasChoices("NRC_INDEX_DIR"))
    audit_dir: Path = Field(default=PROJECT_ROOT / "audit", validation_alias=AliasChoices("NRC_AUDIT_DIR"))

    # ------------------------------------------------------- embeddings / retrieval
    embedding_model: str = Field(default="jinaai/jina-embeddings-v2-base-en", validation_alias=AliasChoices("NRC_EMBEDDING_MODEL"))
    embedding_max_seq: int = Field(default=512, validation_alias=AliasChoices("NRC_EMBEDDING_MAX_SEQ"))
    # Cap the model's declared position count. Long-context embedders allocate an
    # attention bias sized to max_position_embeddings at load time; for
    # jina-embeddings-v2 (8192) that tensor alone is ~3.4 GB. Anything above the
    # sequence length we actually use is dead weight. Set 0 to disable the cap.
    embedding_max_positions: int = Field(default=1024, validation_alias=AliasChoices("NRC_EMBEDDING_MAX_POSITIONS"))
    torch_threads: int = Field(default=2, validation_alias=AliasChoices("NRC_TORCH_THREADS"))
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", validation_alias=AliasChoices("NRC_RERANKER_MODEL"))
    enable_reranker: bool = Field(default=True, validation_alias=AliasChoices("NRC_ENABLE_RERANKER"))
    top_k_dense: int = Field(default=30, validation_alias=AliasChoices("NRC_TOP_K_DENSE"))
    top_k_bm25: int = Field(default=30, validation_alias=AliasChoices("NRC_TOP_K_BM25"))
    rerank_candidates: int = Field(default=40, validation_alias=AliasChoices("NRC_RERANK_CANDIDATES"))
    top_k_final: int = Field(default=10, validation_alias=AliasChoices("NRC_TOP_K_FINAL"))
    max_figures_in_context: int = Field(default=4, validation_alias=AliasChoices("NRC_MAX_FIGURES"))
    chunk_tokens: int = Field(default=380, validation_alias=AliasChoices("NRC_CHUNK_TOKENS"))
    chunk_overlap_tokens: int = Field(default=50, validation_alias=AliasChoices("NRC_CHUNK_OVERLAP_TOKENS"))

    # ------------------------------------------------------------------- LLM
    llm_provider: Literal["auto", "anthropic", "openai", "google"] = Field(default="auto", validation_alias=AliasChoices("NRC_LLM_PROVIDER"))

    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-opus-5"
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    # Server-side refusal fallbacks are opt-in: the audit trail must name the exact model
    # that produced every answer, and the installed SDK has no native support for the flag.
    anthropic_enable_fallbacks: bool = False

    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None  # e.g. https://api.core42.ai/v1 (Core42) or a local vLLM/Ollama endpoint
    openai_model: str = "gpt-4o"

    google_api_key: Optional[str] = None
    google_model: str = "gemini-2.5-pro"

    llm_timeout_seconds: float = Field(default=180.0, validation_alias=AliasChoices("NRC_LLM_TIMEOUT"))

    # ------------------------------------------------------------ verification
    fuzzy_threshold: float = Field(default=0.92, validation_alias=AliasChoices("NRC_FUZZY_THRESHOLD"))
    min_quote_words: int = Field(default=3, validation_alias=AliasChoices("NRC_MIN_QUOTE_WORDS"))
    min_quote_chars: int = Field(default=12, validation_alias=AliasChoices("NRC_MIN_QUOTE_CHARS"))
    enable_support_check: bool = Field(default=True, validation_alias=AliasChoices("NRC_ENABLE_SUPPORT_CHECK"))

    # ---------------------------------------------------------------- render
    render_dpi: int = Field(default=110, validation_alias=AliasChoices("NRC_RENDER_DPI"))
    figure_dpi: int = Field(default=150, validation_alias=AliasChoices("NRC_FIGURE_DPI"))

    # ------------------------------------------------------------- helpers
    def resolve(self) -> "Settings":
        """Make relative paths absolute (relative to the project root) and create dirs."""
        for name in ("data_dir", "index_dir", "audit_dir"):
            p = getattr(self, name)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            setattr(self, name, p)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        return self

    def available_providers(self) -> list[str]:
        out = []
        if self.anthropic_api_key:
            out.append("anthropic")
        if self.openai_api_key:
            out.append("openai")
        if self.google_api_key:
            out.append("google")
        return out

    def selected_provider(self) -> Optional[str]:
        if self.llm_provider != "auto":
            return self.llm_provider
        avail = self.available_providers()
        return avail[0] if avail else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings().resolve()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
