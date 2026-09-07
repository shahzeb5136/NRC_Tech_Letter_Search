"""Token-aware chunking with exact page provenance.

Text chunks are *verbatim slices* of the page text (``page.text[char_start:char_end]``),
never cross a page boundary and record the union bounding box of the blocks they
come from. That property is what lets the verifier prove a quoted passage exists on
a given page and lets the renderer highlight it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from typing import Callable, Optional

from nrc_rag.ingest.pdf_extract import DocumentData, FigureData, PageData, TableData

log = logging.getLogger(__name__)

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"“])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page_number: int
    kind: str  # text | table | figure
    section: str
    section_path: str
    text: str
    char_start: int
    char_end: int
    bbox: Optional[list[float]]
    figure_id: Optional[str]
    token_count: int

    def to_dict(self) -> dict:
        return asdict(self)


class TokenCounter:
    """tiktoken cl100k_base when available, otherwise a word-based estimate."""

    def __init__(self) -> None:
        self._enc = None
        try:
            import tiktoken

            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # pragma: no cover - offline fallback
            log.warning("tiktoken unavailable (%s); using word-count estimate", exc)

    def count(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text, disallowed_special=()))
        return int(len(text.split()) * 1.3) + 1


def _sentence_spans(page_text: str, page: PageData) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for b in page.blocks:
        btext = page_text[b.char_start : b.char_end]
        last = 0
        for m in SENT_SPLIT_RE.finditer(btext):
            end = m.start()
            if end > last:
                spans.append((b.char_start + last, b.char_start + end))
            last = m.end()
        if last < len(btext):
            spans.append((b.char_start + last, b.char_end))
    return [(s, e) for s, e in spans if page_text[s:e].strip()]


def _split_long_span(page_text: str, start: int, end: int, counter: TokenCounter, target: int) -> list[tuple[int, int]]:
    text = page_text[start:end]
    n_tokens = counter.count(text)
    if n_tokens <= target:
        return [(start, end)]
    parts = max(2, -(-n_tokens // target))
    approx = len(text) // parts
    out = []
    pos = 0
    for i in range(parts):
        if i == parts - 1:
            cut = len(text)
        else:
            cut = text.rfind(" ", pos + int(approx * 0.6), pos + approx + int(approx * 0.4))
            if cut <= pos:
                cut = min(len(text), pos + approx)
        if cut > pos:
            out.append((start + pos, start + cut))
        pos = cut
    return out


def _bbox_for_span(page: PageData, start: int, end: int) -> Optional[list[float]]:
    rect = None
    for b in page.blocks:
        if b.char_end <= start or b.char_start >= end:
            continue
        if rect is None:
            rect = [b.x0, b.y0, b.x1, b.y1]
        else:
            rect = [min(rect[0], b.x0), min(rect[1], b.y0), max(rect[2], b.x1), max(rect[3], b.y1)]
    return rect


def _section_at(page: PageData, y: float) -> tuple[str, str]:
    lookup: Optional[Callable] = getattr(page, "_section_lookup", None)
    if lookup is not None:
        try:
            s, p = lookup(y)
            if s:
                return s, p
        except Exception:
            pass
    return page.section, page.section_path


def chunk_page(page: PageData, counter: TokenCounter, target_tokens: int = 380, overlap_tokens: int = 50) -> list[Chunk]:
    chunks: list[Chunk] = []
    if page.is_toc or not page.text.strip():
        return chunks

    spans: list[tuple[int, int]] = []
    for s, e in _sentence_spans(page.text, page):
        spans.extend(_split_long_span(page.text, s, e, counter, target_tokens))

    idx = 0
    i = 0
    while i < len(spans):
        start = spans[i][0]
        end = spans[i][1]
        tokens = counter.count(page.text[start:end])
        j = i + 1
        while j < len(spans):
            t = counter.count(page.text[spans[j][0] : spans[j][1]])
            if tokens + t > target_tokens:
                break
            tokens += t
            end = spans[j][1]
            j += 1
        text = page.text[start:end]
        if text.strip():
            idx += 1
            bbox = _bbox_for_span(page, start, end)
            section, section_path = _section_at(page, bbox[1] if bbox else 0.0)
            chunks.append(
                Chunk(
                    chunk_id=f"{page.doc_id}:p{page.page_number}:c{idx}",
                    doc_id=page.doc_id,
                    page_number=page.page_number,
                    kind="text",
                    section=section,
                    section_path=section_path,
                    text=text,
                    char_start=start,
                    char_end=end,
                    bbox=bbox,
                    figure_id=None,
                    token_count=tokens,
                )
            )
        if j >= len(spans):
            break
        # overlap: step back over trailing sentences worth up to overlap_tokens
        back = j
        acc = 0
        while back - 1 > i:
            t = counter.count(page.text[spans[back - 1][0] : spans[back - 1][1]])
            if acc + t > overlap_tokens:
                break
            acc += t
            back -= 1
        i = back if back > i else j
    return chunks


def table_chunk(page: PageData, table: TableData, n: int, doc: DocumentData, counter: TokenCounter) -> Chunk:
    header = f"Table on page {page.page_number} of {doc.doc_id}"
    if table.caption:
        header += f" - {table.caption}"
    text = f"{header}\n\n{table.markdown}"
    section, section_path = _section_at(page, table.bbox[1])
    return Chunk(
        chunk_id=f"{doc.doc_id}:p{page.page_number}:t{n}",
        doc_id=doc.doc_id,
        page_number=page.page_number,
        kind="table",
        section=section,
        section_path=section_path,
        text=text,
        char_start=-1,
        char_end=-1,
        bbox=list(table.bbox),
        figure_id=None,
        token_count=counter.count(text),
    )


def figure_chunk_text(doc_id: str, tlr: str, page_number: int, caption: str, nearby_text: str, description: str) -> str:
    parts = [f"Figure on page {page_number} of {doc_id}" + (f" ({tlr})" if tlr else "")]
    if caption:
        parts[0] += f": {caption}"
    if nearby_text:
        parts.append(f"Context: {nearby_text}")
    if description:
        parts.append(f"Description (AI-generated, verify against the image): {description}")
    return "\n\n".join(parts)


def figure_chunk(page: PageData, fig: FigureData, doc: DocumentData, counter: TokenCounter) -> Chunk:
    text = figure_chunk_text(doc.doc_id, doc.tlr_number, page.page_number, fig.caption, fig.nearby_text, fig.description)
    section, section_path = _section_at(page, fig.bbox[1])
    return Chunk(
        chunk_id=fig.figure_id,
        doc_id=doc.doc_id,
        page_number=page.page_number,
        kind="figure",
        section=section,
        section_path=section_path,
        text=text,
        char_start=-1,
        char_end=-1,
        bbox=list(fig.bbox),
        figure_id=fig.figure_id,
        token_count=counter.count(text),
    )


def chunk_document(doc: DocumentData, counter: TokenCounter, target_tokens: int = 380, overlap_tokens: int = 50) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in doc.pages:
        chunks.extend(chunk_page(page, counter, target_tokens, overlap_tokens))
        for n, table in enumerate(page.tables, start=1):
            chunks.append(table_chunk(page, table, n, doc, counter))
        for fig in page.figures:
            chunks.append(figure_chunk(page, fig, doc, counter))
    return chunks
