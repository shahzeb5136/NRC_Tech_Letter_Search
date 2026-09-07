"""Portable dense-vector store: one ``.npz`` file plus exact cosine search.

Why not a vector database. At this corpus size (~3k passages) an exact dot
product over a normalised matrix is microseconds, so an approximate-nearest-
neighbour index buys nothing but costs a dependency, ~30 MB of on-disk overhead
and a portability problem: a database's binary index files are written by a
specific library version on a specific machine, and a prebuilt one committed to
a repository may not open on the host that checks it out. A ``.npz`` of float32
vectors is just numbers, so it restores identically anywhere and can be shipped
with the code.

Vectors are stored L2-normalised, which makes cosine similarity a dot product.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

log = logging.getLogger(__name__)

VECTORS_FILE = "vectors.npz"


class VectorStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self.ids: list[str] = []
        self.doc_ids: list[str] = []
        self.kinds: list[str] = []
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._pos: dict[str, int] = {}
        self.load()

    # ------------------------------------------------------------------ io
    def load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self.ids, self.doc_ids, self.kinds = [], [], []
                self.matrix = np.zeros((0, 0), dtype=np.float32)
                self._pos = {}
                return
            try:
                with np.load(self.path, allow_pickle=False) as z:
                    self.matrix = np.asarray(z["vectors"], dtype=np.float32)
                    self.ids = [str(x) for x in z["ids"]]
                    self.doc_ids = [str(x) for x in z["doc_ids"]]
                    self.kinds = [str(x) for x in z["kinds"]]
            except Exception as exc:  # pragma: no cover - corrupt file
                log.error("Could not read %s (%s); starting empty", self.path, exc)
                self.ids, self.doc_ids, self.kinds = [], [], []
                self.matrix = np.zeros((0, 0), dtype=np.float32)
            self._reindex()

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Write through a handle: np.savez appends ".npz" to a *path* that does
            # not already end in it, which would leave the temp file misnamed.
            tmp = self.path.with_name(self.path.name + ".tmp")
            with open(tmp, "wb") as fh:
                np.savez(
                    fh,
                    vectors=self.matrix.astype(np.float32, copy=False),
                    ids=np.array(self.ids, dtype=object).astype("U"),
                    doc_ids=np.array(self.doc_ids, dtype=object).astype("U"),
                    kinds=np.array(self.kinds, dtype=object).astype("U"),
                )
            tmp.replace(self.path)

    def _reindex(self) -> None:
        self._pos = {cid: i for i, cid in enumerate(self.ids)}

    # -------------------------------------------------------------- mutate
    def upsert(self, chunk_ids: list[str], doc_ids: list[str], kinds: list[str], vectors: np.ndarray) -> None:
        """Add or replace vectors. Existing ids keep their row; new ones are appended."""
        if not chunk_ids:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        with self._lock:
            if self.matrix.size == 0:
                self.matrix = np.zeros((0, vectors.shape[1]), dtype=np.float32)
            if self.matrix.shape[1] != vectors.shape[1]:
                raise ValueError(f"embedding dimension changed ({self.matrix.shape[1]} -> {vectors.shape[1]}); rebuild the index with --force")

            new_rows, new_ids, new_docs, new_kinds = [], [], [], []
            for cid, did, kind, vec in zip(chunk_ids, doc_ids, kinds, vectors):
                pos = self._pos.get(cid)
                if pos is None:
                    new_rows.append(vec)
                    new_ids.append(cid)
                    new_docs.append(did)
                    new_kinds.append(kind)
                else:
                    self.matrix[pos] = vec
                    self.doc_ids[pos] = did
                    self.kinds[pos] = kind
            if new_rows:
                self.matrix = np.vstack([self.matrix, np.asarray(new_rows, dtype=np.float32)]) if self.matrix.size else np.asarray(new_rows, dtype=np.float32)
                self.ids.extend(new_ids)
                self.doc_ids.extend(new_docs)
                self.kinds.extend(new_kinds)
                self._reindex()

    def delete_ids(self, chunk_ids: Iterable[str]) -> None:
        drop = {c for c in chunk_ids if c in self._pos}
        if not drop:
            return
        with self._lock:
            keep = [i for i, cid in enumerate(self.ids) if cid not in drop]
            self.matrix = self.matrix[keep] if self.matrix.size else self.matrix
            self.ids = [self.ids[i] for i in keep]
            self.doc_ids = [self.doc_ids[i] for i in keep]
            self.kinds = [self.kinds[i] for i in keep]
            self._reindex()

    def delete_doc(self, doc_id: str) -> None:
        with self._lock:
            keep = [i for i, d in enumerate(self.doc_ids) if d != doc_id]
            if len(keep) == len(self.ids):
                return
            self.matrix = self.matrix[keep] if self.matrix.size else self.matrix
            self.ids = [self.ids[i] for i in keep]
            self.doc_ids = [self.doc_ids[i] for i in keep]
            self.kinds = [self.kinds[i] for i in keep]
            self._reindex()

    # --------------------------------------------------------------- query
    def query(self, vector: list[float] | np.ndarray, n: int, doc_ids: Optional[set[str]] = None, kinds: Optional[set[str]] = None) -> list[tuple[str, float]]:
        with self._lock:
            if self.matrix.size == 0 or n <= 0:
                return []
            q = np.asarray(vector, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(q))
            if norm == 0:
                return []
            q = q / norm

            mask = None
            if doc_ids is not None:
                mask = np.fromiter((d in doc_ids for d in self.doc_ids), dtype=bool, count=len(self.doc_ids))
            if kinds is not None:
                km = np.fromiter((k in kinds for k in self.kinds), dtype=bool, count=len(self.kinds))
                mask = km if mask is None else (mask & km)

            scores = self.matrix @ q
            if mask is not None:
                idx_pool = np.flatnonzero(mask)
                if idx_pool.size == 0:
                    return []
                sub = scores[idx_pool]
                take = min(n, sub.size)
                top = idx_pool[np.argpartition(-sub, take - 1)[:take]]
            else:
                take = min(n, scores.size)
                top = np.argpartition(-scores, take - 1)[:take]
            top = top[np.argsort(-scores[top])]
            return [(self.ids[i], float(scores[i])) for i in top]

    def count(self) -> int:
        return len(self.ids)

    def stats(self) -> dict:
        return {"vectors": len(self.ids), "dim": int(self.matrix.shape[1]) if self.matrix.size else 0, "bytes": int(self.matrix.nbytes)}


def export_manifest(store: VectorStore) -> str:
    return json.dumps(store.stats(), sort_keys=True)
