"""Cached resources shared by the UI pages."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

import streamlit as st

from nrc_rag.audit.log import AuditLog
from nrc_rag.config import PROJECT_ROOT, Settings, get_settings, reset_settings_cache
from nrc_rag.index.embeddings import Embedder, Reranker, try_load_reranker
from nrc_rag.index.retriever import HybridRetriever
from nrc_rag.index.store import IndexStore
from nrc_rag.llm import get_provider
from nrc_rag.llm.base import LLMProvider
from nrc_rag.render.figures import figure_png
from nrc_rag.render.page_render import render_page, render_region
from nrc_rag.verify.engine import GroundedEngine

SECRET_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "GOOGLE_API_KEY", "GOOGLE_MODEL", "ANTHROPIC_MODEL", "NRC_LLM_PROVIDER")


def load_streamlit_secrets() -> None:
    """Allow keys in .streamlit/secrets.toml (e.g. Streamlit Cloud) without overriding the environment."""
    candidates = [PROJECT_ROOT / ".streamlit" / "secrets.toml", Path.home() / ".streamlit" / "secrets.toml"]
    if not any(c.exists() for c in candidates):
        return  # touching st.secrets without a file prints warnings in the UI
    try:
        secrets = st.secrets
        for k in SECRET_KEYS:
            if k in secrets and not os.environ.get(k):
                os.environ[k] = str(secrets[k])
    except Exception:
        return


@st.cache_resource(show_spinner=False)
def settings() -> Settings:
    load_streamlit_secrets()
    reset_settings_cache()
    return get_settings()


@st.cache_resource(show_spinner="Opening the local index…")
def store() -> IndexStore:
    return IndexStore(settings().index_dir, data_dir=settings().data_dir)


@st.cache_resource(show_spinner="Loading embedding model (first time only)…")
def embedder() -> Embedder:
    s = settings()
    return Embedder(
        s.embedding_model,
        s.embedding_max_seq,
        max_positions=s.embedding_max_positions or None,
        threads=s.torch_threads or None,
    )


@st.cache_resource(show_spinner="Loading re-ranker…")
def reranker() -> Optional[Reranker]:
    s = settings()
    if not s.enable_reranker:
        return None
    return try_load_reranker(s.reranker_model)


@st.cache_resource(show_spinner="Building lexical index…")
def retriever() -> HybridRetriever:
    r = HybridRetriever(store(), embedder(), settings(), reranker())
    r.refresh()
    return r


def retrieval_status() -> Optional[str]:
    """``None`` when search is usable, otherwise why it is not.

    The embedding stack (torch, sentence-transformers) is the heaviest thing here
    and the most likely to be missing or unloadable on a constrained host. When it
    is, the rest of the app - the document library, the audit trail, the
    methodology - is still perfectly usable, so this reports the problem instead of
    letting an import error take the whole page down.
    """
    try:
        retriever()
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


@st.cache_resource(show_spinner=False)
def audit() -> AuditLog:
    return AuditLog(settings().audit_dir)


@st.cache_resource(show_spinner=False)
def provider(name: str) -> Optional[LLMProvider]:
    try:
        return get_provider(settings(), name)
    except Exception as exc:  # pragma: no cover
        st.warning(f"Provider {name} could not be initialised: {exc}")
        return None


def engine(provider_name: Optional[str]) -> GroundedEngine:
    p = provider(provider_name) if provider_name else None
    eng = GroundedEngine(store(), retriever(), settings(), audit(), p)
    return eng


@st.cache_data(show_spinner=False, max_entries=200)
def page_png(pdf_path: str, page_number: int, rects: tuple, dpi: int, approximate: bool) -> bytes:
    return render_page(pdf_path, page_number, [list(r) for r in rects], dpi=dpi, approximate=approximate)


@st.cache_data(show_spinner=False, max_entries=200)
def region_png(pdf_path: str, page_number: int, bbox: tuple, dpi: int) -> bytes:
    return render_region(pdf_path, page_number, list(bbox), dpi=dpi)


@st.cache_data(show_spinner=False, max_entries=120)
def figure_image(figure_id: Optional[str]) -> Optional[bytes]:
    """Figure PNG: the cached file when ingestion left one, otherwise cropped
    from the source PDF. Lets a deployment ship the index without the images."""
    if not figure_id:
        return None
    return figure_png(store(), figure_id, settings().figure_dpi)


@st.cache_data(show_spinner=False)
def logo_b64() -> Optional[str]:
    p = PROJECT_ROOT / "logo.png"
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode("ascii")
    return None


def index_ready() -> bool:
    try:
        return store().stats()["chunks"] > 0
    except Exception:
        return False


def doc_label(d) -> str:
    t = d.tlr_number or d.doc_id
    title = d.title if len(d.title) <= 70 else d.title[:67] + "…"
    return f"{t} — {title}"


def open_in_system_viewer(path: str) -> tuple[bool, str]:
    try:
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
            return True, "Opened in the system PDF viewer."
        import subprocess

        subprocess.Popen(["xdg-open" if os.uname().sysname != "Darwin" else "open", path])
        return True, "Opened in the system PDF viewer."
    except Exception as exc:
        return False, f"Could not open the file: {exc}"


def file_bytes(path: str) -> Optional[bytes]:
    try:
        return Path(path).read_bytes()
    except Exception:
        return None
