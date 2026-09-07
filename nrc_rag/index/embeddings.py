"""Local embedding and reranking models (sentence-transformers, CPU friendly).

Embeddings are computed locally so that document text never leaves the machine
during indexing and retrieval; only the retrieved excerpts sent to the configured
LLM provider leave the machine, and the audit trail records exactly which ones.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str, max_seq_length: int = 1024, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        self.model.max_seq_length = max_seq_length
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int = 16, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, query: str) -> list[float]:
        return self.encode([query])[0].tolist()


class Reranker:
    """Cross-encoder reranker; optional, loaded lazily."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict([(query, t) for t in texts], show_progress_bar=False)
        return [float(s) for s in np.asarray(scores).reshape(-1)]


def try_load_reranker(model_name: str) -> Optional[Reranker]:
    try:
        return Reranker(model_name)
    except Exception as exc:  # pragma: no cover - depends on network / cache
        log.warning("Reranker %s unavailable: %s", model_name, exc)
        return None
