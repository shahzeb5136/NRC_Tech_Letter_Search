"""Hybrid retrieval: dense (ChromaDB) + lexical (BM25) fused with reciprocal-rank
fusion, optionally re-scored by a cross-encoder.

Every retrieved chunk keeps its component scores so the audit trail can show
*why* a passage was in the model's context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from rank_bm25 import BM25Okapi

from nrc_rag.config import Settings
from nrc_rag.index.embeddings import Embedder, Reranker
from nrc_rag.index.store import ChunkRow, IndexStore
from nrc_rag.utils import tokenize_for_bm25

log = logging.getLogger(__name__)

RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk: ChunkRow
    score: float
    rank: int
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    doc_title: str = ""
    tlr_number: str = ""

    def to_audit(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "page": self.chunk.page_number,
            "kind": self.chunk.kind,
            "section": self.chunk.section,
            "rank": self.rank,
            "score": round(self.score, 6),
            "dense_score": None if self.dense_score is None else round(self.dense_score, 6),
            "dense_rank": self.dense_rank,
            "bm25_score": None if self.bm25_score is None else round(self.bm25_score, 6),
            "bm25_rank": self.bm25_rank,
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
        }


@dataclass
class _Lexical:
    ids: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    bm25: Optional[BM25Okapi] = None
    n_chunks: int = -1


class HybridRetriever:
    def __init__(self, store: IndexStore, embedder: Embedder, settings: Settings, reranker: Optional[Reranker] = None) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker
        self._lex = _Lexical()
        self._doc_meta: dict[str, tuple[str, str]] = {}

    # ------------------------------------------------------------- lexical
    def _ensure_lexical(self) -> None:
        n = self.store.stats()["chunks"]
        if self._lex.bm25 is not None and self._lex.n_chunks == n:
            return
        ids, doc_ids, kinds, corpus = [], [], [], []
        for c in self.store.iter_chunks():
            ids.append(c.chunk_id)
            doc_ids.append(c.doc_id)
            kinds.append(c.kind)
            corpus.append(tokenize_for_bm25(c.text))
        self._lex = _Lexical(ids=ids, doc_ids=doc_ids, kinds=kinds, bm25=BM25Okapi(corpus) if corpus else None, n_chunks=n)
        self._doc_meta = {d.doc_id: (d.title, d.tlr_number) for d in self.store.list_documents()}
        log.info("BM25 index built over %d chunks", n)

    def refresh(self) -> None:
        self._lex = _Lexical()
        self._ensure_lexical()

    # -------------------------------------------------------------- search
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_ids: Optional[list[str]] = None,
        kinds: Optional[list[str]] = None,
        use_reranker: bool = True,
    ) -> list[RetrievedChunk]:
        self._ensure_lexical()
        top_k = top_k or self.settings.top_k_final
        doc_filter = set(doc_ids) if doc_ids else None
        kind_filter = set(kinds) if kinds else None

        # dense
        where_clauses = []
        if doc_filter:
            where_clauses.append({"doc_id": {"$in": sorted(doc_filter)}})
        if kind_filter:
            where_clauses.append({"kind": {"$in": sorted(kind_filter)}})
        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}
        qvec = self.embedder.encode_query(query)
        dense = self.store.query_vectors(qvec, self.settings.top_k_dense, where)
        dense_rank = {cid: (i + 1, s) for i, (cid, s) in enumerate(dense)}

        # lexical
        bm25_rank: dict[str, tuple[int, float]] = {}
        if self._lex.bm25 is not None:
            scores = self._lex.bm25.get_scores(tokenize_for_bm25(query))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            r = 0
            for i in order:
                if scores[i] <= 0:
                    break
                if doc_filter and self._lex.doc_ids[i] not in doc_filter:
                    continue
                if kind_filter and self._lex.kinds[i] not in kind_filter:
                    continue
                r += 1
                bm25_rank[self._lex.ids[i]] = (r, float(scores[i]))
                if r >= self.settings.top_k_bm25:
                    break

        # reciprocal rank fusion
        fused: dict[str, float] = {}
        for cid, (rk, _) in dense_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rk)
        for cid, (rk, _) in bm25_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rk)
        if not fused:
            return []
        candidates = sorted(fused.items(), key=lambda kv: -kv[1])[: max(top_k, self.settings.rerank_candidates)]
        rows = self.store.get_chunks([cid for cid, _ in candidates])

        results: list[RetrievedChunk] = []
        for cid, fscore in candidates:
            row = rows.get(cid)
            if row is None:
                continue
            title, tlr = self._doc_meta.get(row.doc_id, ("", ""))
            d = dense_rank.get(cid)
            b = bm25_rank.get(cid)
            results.append(
                RetrievedChunk(
                    chunk=row, score=fscore, rank=0,
                    dense_score=d[1] if d else None, dense_rank=d[0] if d else None,
                    bm25_score=b[1] if b else None, bm25_rank=b[0] if b else None,
                    doc_title=title, tlr_number=tlr,
                )
            )

        if use_reranker and self.reranker is not None and results:
            texts = [r.chunk.text[:2000] for r in results]
            try:
                rs = self.reranker.score(query, texts)
                for r, s in zip(results, rs):
                    r.rerank_score = s
                results.sort(key=lambda r: -(r.rerank_score or -1e9))
            except Exception as exc:  # pragma: no cover
                log.warning("rerank failed, using fused order: %s", exc)
                results.sort(key=lambda r: -r.score)
        else:
            results.sort(key=lambda r: -r.score)

        final = results[:top_k]
        for i, r in enumerate(final, start=1):
            r.rank = i
        return final
