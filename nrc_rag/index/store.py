"""Persistent local index.

* ``catalog.db`` (SQLite) - documents, page text with block geometry, chunks, figures.
  This is the source of truth the verifier checks quotes against.
* ``vectors.npz`` - dense embeddings for the chunks, keyed by chunk_id. A plain
  array rather than a vector database: see ``nrc_rag/index/vectors.py``.
* ``manifest.json`` - what was indexed, with file hashes and pipeline versions.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from nrc_rag.config import PIPELINE_VERSION
from nrc_rag.index.vectors import VECTORS_FILE, VectorStore
from nrc_rag.ingest.chunker import Chunk
from nrc_rag.ingest.pdf_extract import DocumentData
from nrc_rag.utils import utc_now_iso

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    title TEXT,
    tlr_number TEXT,
    report_date TEXT,
    organization TEXT,
    page_count INTEGER,
    toc_json TEXT,
    ingested_at TEXT,
    pipeline_version TEXT,
    embedding_model TEXT
);
CREATE TABLE IF NOT EXISTS pages (
    doc_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    label TEXT,
    width REAL,
    height REAL,
    text TEXT,
    section TEXT,
    section_path TEXT,
    is_toc INTEGER,
    blocks_json TEXT,
    PRIMARY KEY (doc_id, page_number)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    kind TEXT NOT NULL,
    section TEXT,
    section_path TEXT,
    text TEXT NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    bbox_json TEXT,
    figure_id TEXT,
    token_count INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS figures (
    figure_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    bbox_json TEXT,
    caption TEXT,
    image_path TEXT,
    sha1 TEXT,
    source TEXT,
    nearby_text TEXT,
    description TEXT,
    description_model TEXT,
    described_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_figures_doc ON figures(doc_id);
"""


@dataclass
class DocumentRow:
    doc_id: str
    path: str
    sha256: str
    title: str
    tlr_number: str
    report_date: str
    organization: str
    page_count: int
    toc: list
    ingested_at: str
    pipeline_version: str
    embedding_model: str
    chunk_count: int = 0
    figure_count: int = 0


@dataclass
class ChunkRow:
    chunk_id: str
    doc_id: str
    page_number: int
    kind: str
    section: str
    section_path: str
    text: str
    char_start: int
    char_end: int
    bbox: Optional[list[float]]
    figure_id: Optional[str]
    token_count: int


@dataclass
class PageRow:
    doc_id: str
    page_number: int
    label: str
    width: float
    height: float
    text: str
    section: str
    section_path: str
    is_toc: bool
    blocks: list[dict]


@dataclass
class FigureRow:
    figure_id: str
    doc_id: str
    page_number: int
    bbox: list[float]
    caption: str
    image_path: str
    sha1: str
    source: str
    nearby_text: str
    description: str
    description_model: str
    described_at: str


class IndexStore:
    def __init__(self, index_dir: Path, embedding_dim: Optional[int] = None, data_dir: Optional[Path] = None) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.index_dir / "catalog.db"
        self.figures_dir = self.index_dir / "figures"
        self.manifest_path = self.index_dir / "manifest.json"
        # An index is portable: it may have been built on another machine, so the
        # absolute PDF paths recorded at build time will not resolve here. Source
        # files are re-found by accession number under the data directory.
        self.data_dir = Path(data_dir) if data_dir else self.index_dir.parent / "Data"
        self._path_cache: dict[str, str] = {}
        self._scanned = False
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=DELETE")
        self.conn.executescript(SCHEMA)
        self._vectors: Optional[VectorStore] = None
        self._embedding_dim = embedding_dim

    # ------------------------------------------------------------ source paths
    def _scan_data_dir(self) -> None:
        if self._scanned:
            return
        self._scanned = True
        try:
            for p in self.data_dir.rglob("*.pdf"):
                self._path_cache.setdefault(p.stem, str(p))
        except Exception as exc:  # pragma: no cover
            log.warning("could not scan %s for source PDFs: %s", self.data_dir, exc)

    def resolve_doc_path(self, doc_id: str, stored_path: str) -> str:
        """The usable path to a document's PDF on *this* machine."""
        if stored_path and Path(stored_path).exists():
            return stored_path
        self._scan_data_dir()
        return self._path_cache.get(doc_id, stored_path)

    # ---------------------------------------------------------------- vectors
    @property
    def vectors(self) -> VectorStore:
        if self._vectors is None:
            self._vectors = VectorStore(self.index_dir / VECTORS_FILE)
        return self._vectors

    # --------------------------------------------------------------- manifest
    def read_manifest(self) -> dict:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def write_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # ------------------------------------------------------------- documents
    def list_documents(self) -> list[DocumentRow]:
        cur = self.conn.execute(
            """
            SELECT d.doc_id, d.path, d.sha256, d.title, d.tlr_number, d.report_date, d.organization, d.page_count,
                   d.toc_json, d.ingested_at, d.pipeline_version, d.embedding_model,
                   (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.doc_id),
                   (SELECT COUNT(*) FROM figures f WHERE f.doc_id = d.doc_id)
            FROM documents d ORDER BY d.doc_id
            """
        )
        rows = []
        for r in cur.fetchall():
            rows.append(
                DocumentRow(
                    doc_id=r[0], path=self.resolve_doc_path(r[0], r[1]), sha256=r[2], title=r[3] or "", tlr_number=r[4] or "", report_date=r[5] or "",
                    organization=r[6] or "", page_count=r[7] or 0, toc=json.loads(r[8] or "[]"), ingested_at=r[9] or "",
                    pipeline_version=r[10] or "", embedding_model=r[11] or "", chunk_count=r[12], figure_count=r[13],
                )
            )
        return rows

    def get_document(self, doc_id: str) -> Optional[DocumentRow]:
        for d in self.list_documents():
            if d.doc_id == doc_id:
                return d
        return None

    def delete_document(self, doc_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            self.conn.execute("DELETE FROM pages WHERE doc_id=?", (doc_id,))
            self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self.conn.execute("DELETE FROM figures WHERE doc_id=?", (doc_id,))
        self.vectors.delete_doc(doc_id)
        self.vectors.save()

    def upsert_document(self, doc: DocumentData, chunks: list[Chunk], embedding_model: str) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO documents
                   (doc_id, path, sha256, title, tlr_number, report_date, organization, page_count, toc_json,
                    ingested_at, pipeline_version, embedding_model)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    doc.doc_id, doc.path, doc.sha256, doc.title, doc.tlr_number, doc.report_date, doc.organization,
                    doc.page_count, json.dumps(doc.toc), utc_now_iso(), PIPELINE_VERSION, embedding_model,
                ),
            )
            self.conn.execute("DELETE FROM pages WHERE doc_id=?", (doc.doc_id,))
            self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc.doc_id,))
            self.conn.execute("DELETE FROM figures WHERE doc_id=?", (doc.doc_id,))
            self.conn.executemany(
                """INSERT INTO pages (doc_id, page_number, label, width, height, text, section, section_path, is_toc, blocks_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        p.doc_id, p.page_number, p.label, p.width, p.height, p.text, p.section, p.section_path,
                        1 if p.is_toc else 0,
                        json.dumps([[b.x0, b.y0, b.x1, b.y1, b.char_start, b.char_end] for b in p.blocks]),
                    )
                    for p in doc.pages
                ],
            )
            self.conn.executemany(
                """INSERT INTO chunks (chunk_id, doc_id, page_number, kind, section, section_path, text, char_start, char_end,
                   bbox_json, figure_id, token_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        c.chunk_id, c.doc_id, c.page_number, c.kind, c.section, c.section_path, c.text, c.char_start,
                        c.char_end, json.dumps(c.bbox) if c.bbox else None, c.figure_id, c.token_count,
                    )
                    for c in chunks
                ],
            )
            self.conn.executemany(
                """INSERT INTO figures (figure_id, doc_id, page_number, bbox_json, caption, image_path, sha1, source,
                   nearby_text, description, description_model, described_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        f.figure_id, f.doc_id, f.page_number, json.dumps(f.bbox), f.caption, f.image_path, f.sha1, f.source,
                        f.nearby_text, f.description, "", "",
                    )
                    for p in doc.pages
                    for f in p.figures
                ],
            )

    # ------------------------------------------------------------------ pages
    def get_page(self, doc_id: str, page_number: int) -> Optional[PageRow]:
        r = self.conn.execute(
            "SELECT doc_id, page_number, label, width, height, text, section, section_path, is_toc, blocks_json FROM pages WHERE doc_id=? AND page_number=?",
            (doc_id, page_number),
        ).fetchone()
        if not r:
            return None
        blocks = [
            {"bbox": b[:4], "char_start": b[4], "char_end": b[5]} for b in json.loads(r[9] or "[]")
        ]
        return PageRow(r[0], r[1], r[2] or "", r[3], r[4], r[5] or "", r[6] or "", r[7] or "", bool(r[8]), blocks)

    # ----------------------------------------------------------------- chunks
    @staticmethod
    def _chunk_row(r: tuple) -> ChunkRow:
        return ChunkRow(
            chunk_id=r[0], doc_id=r[1], page_number=r[2], kind=r[3], section=r[4] or "", section_path=r[5] or "",
            text=r[6], char_start=r[7], char_end=r[8], bbox=json.loads(r[9]) if r[9] else None, figure_id=r[10],
            token_count=r[11] or 0,
        )

    _CHUNK_COLS = "chunk_id, doc_id, page_number, kind, section, section_path, text, char_start, char_end, bbox_json, figure_id, token_count"

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRow]:
        r = self.conn.execute(f"SELECT {self._CHUNK_COLS} FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        return self._chunk_row(r) if r else None

    def get_chunks(self, chunk_ids: Iterable[str]) -> dict[str, ChunkRow]:
        ids = list(dict.fromkeys(chunk_ids))
        out: dict[str, ChunkRow] = {}
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            q = f"SELECT {self._CHUNK_COLS} FROM chunks WHERE chunk_id IN ({','.join('?' * len(batch))})"
            for r in self.conn.execute(q, batch).fetchall():
                row = self._chunk_row(r)
                out[row.chunk_id] = row
        return out

    def iter_chunks(self, doc_ids: Optional[list[str]] = None) -> Iterable[ChunkRow]:
        if doc_ids:
            q = f"SELECT {self._CHUNK_COLS} FROM chunks WHERE doc_id IN ({','.join('?' * len(doc_ids))}) ORDER BY doc_id, page_number, chunk_id"
            cur = self.conn.execute(q, doc_ids)
        else:
            cur = self.conn.execute(f"SELECT {self._CHUNK_COLS} FROM chunks ORDER BY doc_id, page_number, chunk_id")
        for r in cur:
            yield self._chunk_row(r)

    def update_chunk_text(self, chunk_id: str, text: str, token_count: int) -> None:
        with self.conn:
            self.conn.execute("UPDATE chunks SET text=?, token_count=? WHERE chunk_id=?", (text, token_count, chunk_id))

    # ---------------------------------------------------------------- figures
    @staticmethod
    def _figure_row(r: tuple) -> FigureRow:
        return FigureRow(
            figure_id=r[0], doc_id=r[1], page_number=r[2], bbox=json.loads(r[3] or "[0,0,0,0]"), caption=r[4] or "",
            image_path=r[5] or "", sha1=r[6] or "", source=r[7] or "", nearby_text=r[8] or "", description=r[9] or "",
            description_model=r[10] or "", described_at=r[11] or "",
        )

    _FIG_COLS = "figure_id, doc_id, page_number, bbox_json, caption, image_path, sha1, source, nearby_text, description, description_model, described_at"

    def get_figure(self, figure_id: str) -> Optional[FigureRow]:
        r = self.conn.execute(f"SELECT {self._FIG_COLS} FROM figures WHERE figure_id=?", (figure_id,)).fetchone()
        return self._figure_row(r) if r else None

    def list_figures(self, doc_id: Optional[str] = None, undescribed_only: bool = False) -> list[FigureRow]:
        q = f"SELECT {self._FIG_COLS} FROM figures"
        conds, params = [], []
        if doc_id:
            conds.append("doc_id=?")
            params.append(doc_id)
        if undescribed_only:
            conds.append("(description IS NULL OR description='')")
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY doc_id, page_number, figure_id"
        return [self._figure_row(r) for r in self.conn.execute(q, params).fetchall()]

    def set_figure_description(self, figure_id: str, description: str, model: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE figures SET description=?, description_model=?, described_at=? WHERE figure_id=?",
                (description, model, utc_now_iso(), figure_id),
            )

    # ---------------------------------------------------------------- vectors
    def add_vectors(self, chunks: list[Chunk | ChunkRow], vectors: np.ndarray, doc_meta: Optional[dict[str, Any]] = None) -> None:
        """Store (or replace) the embeddings for these chunks and persist them."""
        if not chunks:
            return
        self.vectors.upsert(
            [c.chunk_id for c in chunks],
            [c.doc_id for c in chunks],
            [c.kind for c in chunks],
            vectors,
        )
        self.vectors.save()

    def delete_vectors(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.vectors.delete_ids(chunk_ids)
            self.vectors.save()

    def query_vectors(self, vector: list[float], n: int, doc_ids: Optional[set[str]] = None, kinds: Optional[set[str]] = None) -> list[tuple[str, float]]:
        try:
            return self.vectors.query(vector, n, doc_ids=doc_ids, kinds=kinds)
        except Exception as exc:  # pragma: no cover
            log.warning("vector query failed: %s", exc)
            return []

    def vector_count(self) -> int:
        return self.vectors.count()

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        c = self.conn
        out = {
            "documents": c.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            "pages": c.execute("SELECT COUNT(*) FROM pages").fetchone()[0],
            "chunks": c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "text_chunks": c.execute("SELECT COUNT(*) FROM chunks WHERE kind='text'").fetchone()[0],
            "table_chunks": c.execute("SELECT COUNT(*) FROM chunks WHERE kind='table'").fetchone()[0],
            "figures": c.execute("SELECT COUNT(*) FROM figures").fetchone()[0],
            "figures_described": c.execute("SELECT COUNT(*) FROM figures WHERE description<>''").fetchone()[0],
        }
        return out

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
