"""Ingestion pipeline: PDF -> pages/chunks/figures -> local index.

Incremental: a document is re-processed only when its SHA-256, the pipeline
version or the embedding model changed (or ``force=True``). Figure descriptions
that were already produced are preserved across re-indexing (keyed by the
figure image hash) so that paid vision calls are never repeated.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from nrc_rag.config import PIPELINE_VERSION, Settings
from nrc_rag.index.embeddings import Embedder
from nrc_rag.index.store import IndexStore
from nrc_rag.ingest.chunker import Chunk, TokenCounter, chunk_document, figure_chunk_text
from nrc_rag.ingest.pdf_extract import DocumentData, extract_document
from nrc_rag.render.figures import figure_png
from nrc_rag.utils import sha256_file, utc_now_iso

log = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]


@dataclass
class IngestReport:
    processed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    chunks_added: int = 0
    figures_added: int = 0
    elapsed_seconds: float = 0.0


def discover_pdfs(data_dir: Path) -> list[Path]:
    return sorted(p for p in Path(data_dir).rglob("*.pdf") if not p.name.startswith("~$"))


def _needs_reindex(manifest: dict, doc_id: str, sha: str, settings: Settings, force: bool) -> bool:
    if force:
        return True
    entry = (manifest.get("documents") or {}).get(doc_id)
    if not entry:
        return True
    return not (
        entry.get("sha256") == sha
        and entry.get("pipeline_version") == PIPELINE_VERSION
        and entry.get("embedding_model") == settings.embedding_model
        and entry.get("chunk_tokens") == settings.chunk_tokens
    )


def _load_description_cache(store: IndexStore) -> dict[str, dict]:
    path = store.index_dir / "figure_descriptions_cache.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_description_cache(store: IndexStore, cache: dict[str, dict]) -> None:
    path = store.index_dir / "figure_descriptions_cache.json"
    path.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")


def ingest(
    settings: Settings,
    force: bool = False,
    only: Optional[list[str]] = None,
    extract_figures: bool = True,
    progress: Optional[ProgressFn] = None,
    embedder: Optional[Embedder] = None,
) -> IngestReport:
    t0 = time.time()
    say = progress or (lambda msg: log.info(msg))
    report = IngestReport()
    store = IndexStore(settings.index_dir, data_dir=settings.data_dir)
    manifest = store.read_manifest()
    manifest.setdefault("documents", {})
    manifest["pipeline_version"] = PIPELINE_VERSION
    manifest["embedding_model"] = settings.embedding_model
    manifest["chunk_tokens"] = settings.chunk_tokens
    manifest["chunk_overlap_tokens"] = settings.chunk_overlap_tokens
    counter = TokenCounter()
    desc_cache = _load_description_cache(store)

    pdfs = discover_pdfs(settings.data_dir)
    if only:
        wanted = {o.lower().removesuffix(".pdf") for o in only}
        pdfs = [p for p in pdfs if p.stem.lower() in wanted]
    say(f"Found {len(pdfs)} PDF(s) under {settings.data_dir}")

    for pdf in pdfs:
        doc_id = pdf.stem
        try:
            sha = sha256_file(pdf)
            if not _needs_reindex(manifest, doc_id, sha, settings, force):
                report.skipped.append(doc_id)
                say(f"[skip] {doc_id} unchanged")
                continue

            say(f"[extract] {doc_id} ...")
            doc: DocumentData = extract_document(pdf, store.figures_dir, settings.figure_dpi, extract_figures)

            # carry over previously generated figure descriptions (by image hash)
            for page in doc.pages:
                for fig in page.figures:
                    cached = desc_cache.get(fig.sha1)
                    if cached and cached.get("description"):
                        fig.description = cached["description"]

            chunks: list[Chunk] = chunk_document(doc, counter, settings.chunk_tokens, settings.chunk_overlap_tokens)
            say(f"[chunk] {doc_id}: {len(chunks)} chunks, {sum(len(p.figures) for p in doc.pages)} figures, "
                f"{sum(len(p.tables) for p in doc.pages)} tables, title={doc.title!r}")

            old_ids = [c.chunk_id for c in store.iter_chunks([doc_id])]
            if old_ids:
                store.delete_vectors(old_ids)
            store.upsert_document(doc, chunks, settings.embedding_model)
            for page in doc.pages:
                for fig in page.figures:
                    cached = desc_cache.get(fig.sha1)
                    if cached and cached.get("description"):
                        store.set_figure_description(fig.figure_id, cached["description"], cached.get("model", ""))

            if embedder is None:
                say(f"[embed] loading embedding model {settings.embedding_model} ...")
                embedder = Embedder(settings.embedding_model, settings.embedding_max_seq, max_positions=settings.embedding_max_positions or None, threads=settings.torch_threads or None)
            say(f"[embed] {doc_id}: encoding {len(chunks)} chunks ...")
            vectors = embedder.encode([c.text for c in chunks], batch_size=16)
            store.add_vectors(chunks, vectors, {"tlr_number": doc.tlr_number, "title": doc.title})

            manifest["documents"][doc_id] = {
                "sha256": sha,
                "path": str(pdf),
                "title": doc.title,
                "tlr_number": doc.tlr_number,
                "pages": doc.page_count,
                "chunks": len(chunks),
                "figures": sum(len(p.figures) for p in doc.pages),
                "tables": sum(len(p.tables) for p in doc.pages),
                "pipeline_version": PIPELINE_VERSION,
                "embedding_model": settings.embedding_model,
                "chunk_tokens": settings.chunk_tokens,
                "ingested_at": utc_now_iso(),
            }
            manifest["updated_at"] = utc_now_iso()
            store.write_manifest(manifest)
            report.processed.append(doc_id)
            report.chunks_added += len(chunks)
            report.figures_added += sum(len(p.figures) for p in doc.pages)
            say(f"[done] {doc_id}")
        except Exception as exc:  # keep going; report at the end
            log.exception("ingest failed for %s", doc_id)
            report.failed[doc_id] = f"{type(exc).__name__}: {exc}"
            say(f"[FAILED] {doc_id}: {exc}")

    # drop documents whose PDF disappeared
    present = {p.stem for p in discover_pdfs(settings.data_dir)}
    for d in store.list_documents():
        if d.doc_id not in present and not only:
            say(f"[remove] {d.doc_id} (PDF no longer present)")
            store.delete_document(d.doc_id)
            manifest["documents"].pop(d.doc_id, None)
    store.write_manifest(manifest)
    report.elapsed_seconds = time.time() - t0
    store.close()
    return report


def describe_figures(
    settings: Settings,
    provider,
    limit: Optional[int] = None,
    doc_ids: Optional[list[str]] = None,
    progress: Optional[ProgressFn] = None,
    embedder: Optional[Embedder] = None,
    force: bool = False,
) -> int:
    """Generate AI descriptions for figures that do not have one yet.

    The description is stored on the figure, appended to the figure chunk text
    (clearly labelled as AI-generated) and the chunk is re-embedded so that
    questions about diagram *content* can retrieve the figure.
    Returns the number of figures described.
    """
    say = progress or (lambda msg: log.info(msg))
    store = IndexStore(settings.index_dir, data_dir=settings.data_dir)
    cache = _load_description_cache(store)
    figures = store.list_figures(undescribed_only=not force)
    if doc_ids:
        figures = [f for f in figures if f.doc_id in set(doc_ids)]
    if limit:
        figures = figures[:limit]
    say(f"{len(figures)} figure(s) to describe with {provider.name}/{provider.model}")
    done = 0
    counter = TokenCounter()
    docs = {d.doc_id: d for d in store.list_documents()}
    for fig in figures:
        png = figure_png(store, fig.figure_id, settings.figure_dpi)
        if not png:
            say(f"[skip] {fig.figure_id}: no image available")
            continue
        cached = cache.get(fig.sha1)
        if cached and cached.get("description") and not force:
            description, model = cached["description"], cached.get("model", "")
        else:
            try:
                description = provider.describe_figure(png, fig.caption, fig.nearby_text)
                model = f"{provider.name}/{provider.model}"
            except Exception as exc:
                say(f"[FAILED] {fig.figure_id}: {exc}")
                continue
            cache[fig.sha1] = {"description": description, "model": model, "at": utc_now_iso()}
            _save_description_cache(store, cache)
        store.set_figure_description(fig.figure_id, description, model)
        chunk = store.get_chunk(fig.figure_id)
        doc = docs.get(fig.doc_id)
        text = figure_chunk_text(fig.doc_id, doc.tlr_number if doc else "", fig.page_number, fig.caption, fig.nearby_text, description)
        store.update_chunk_text(fig.figure_id, text, counter.count(text))
        if embedder is None:
            embedder = Embedder(settings.embedding_model, settings.embedding_max_seq, max_positions=settings.embedding_max_positions or None, threads=settings.torch_threads or None)
        if chunk is not None:
            chunk.text = text
            vec = embedder.encode([text])
            store.add_vectors([chunk], vec, {"tlr_number": doc.tlr_number if doc else "", "title": doc.title if doc else ""})
        done += 1
        say(f"[described] {fig.figure_id} ({done}/{len(figures)})")
    store.close()
    return done
