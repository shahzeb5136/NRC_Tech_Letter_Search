"""NRC Technical Letter Query System - grounded, citation-verified question answering.

Package layout
--------------
ingest/   PDF extraction (text, tables, figures, outline), chunking, ingestion pipeline
index/    Embeddings, local vector + lexical store, hybrid retrieval
llm/      Provider adapters (Anthropic Claude, OpenAI-compatible, Google Gemini) and prompts
verify/   Deterministic quote verification and the grounded-answer engine
render/   Page rendering with citation highlights
audit/    Append-only audit trail and answer export
"""

from nrc_rag.config import APP_VERSION, PIPELINE_VERSION, Settings, get_settings

__all__ = ["APP_VERSION", "PIPELINE_VERSION", "Settings", "get_settings"]
