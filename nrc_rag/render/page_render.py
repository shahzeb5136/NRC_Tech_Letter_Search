"""Locate quoted text on a PDF page and render the page with highlights.

The rendered page is the human-verifiable proof for a citation: the reader sees
the actual document page with the quoted passage marked. Documents are opened
fresh for every render so highlight drawings never leak between calls.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Optional

import fitz

log = logging.getLogger(__name__)

_LOCK = threading.RLock()


def _search(page: fitz.Page, needle: str) -> list[fitz.Rect]:
    needle = re.sub(r"\s+", " ", needle).strip()
    if not needle:
        return []
    try:
        return list(page.search_for(needle))
    except Exception:
        return []


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def quote_candidates(quote: str, matched_text: str = "") -> list[str]:
    """Text variants to search for on the page, most specific first."""
    out: list[str] = []
    for t in (quote, matched_text):
        t = (t or "").strip().strip("\"'“”‘’")
        if not t:
            continue
        variants = [t, re.sub(r"-\s+", "", re.sub(r"\s+", " ", t))]
        for v in variants:
            if v and v not in out:
                out.append(v)
    return out


def locate_text(pdf_path: str, page_number: int, candidates: list[str], fallback_bbox: Optional[list[float]] = None) -> tuple[list[list[float]], bool]:
    """Return (rects, approximate).

    Tries each candidate text with PyMuPDF's text search; if the whole passage is
    not found (typographic differences, hyphenation), anchors on its first and last
    words and highlights the band between them; finally falls back to the chunk's
    bounding box (marked approximate).
    """
    with _LOCK:
        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            log.warning("cannot open %s: %s", pdf_path, exc)
            return ([fallback_bbox] if fallback_bbox else []), True
        try:
            page = doc[page_number - 1]
            for cand in candidates:
                hits = _search(page, cand)
                if hits:
                    return [list(r) for r in hits], False
            for cand in candidates:
                words = _words(cand)
                if len(words) < 6:
                    continue
                h_hits = _search(page, " ".join(words[:5]))
                t_hits = _search(page, " ".join(words[-5:]))
                if h_hits and t_hits:
                    first = min(h_hits, key=lambda r: (r.y0, r.x0))
                    last = max(t_hits, key=lambda r: (r.y1, r.x1))
                    if last.y1 >= first.y0:
                        rects = [list(first), list(last)]
                        if last.y0 > first.y1 + 1:
                            x0 = min(first.x0, last.x0, fallback_bbox[0] if fallback_bbox else first.x0)
                            x1 = max(first.x1, last.x1, fallback_bbox[2] if fallback_bbox else last.x1)
                            rects.append([x0, first.y1, x1, last.y0])
                        return rects, False
                if h_hits:
                    return [list(r) for r in h_hits], True
            return ([fallback_bbox] if fallback_bbox else []), True
        finally:
            doc.close()


def render_page(pdf_path: str, page_number: int, rects: Optional[list[list[float]]] = None, dpi: int = 110, approximate: bool = False) -> bytes:
    """PNG of the page with translucent highlights (yellow = located quote, dashed amber = approximate region)."""
    with _LOCK:
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_number - 1]
            if rects:
                shape = page.new_shape()
                for r in rects:
                    rect = fitz.Rect(r) & page.rect
                    if rect.is_empty:
                        continue
                    shape.draw_rect(rect)
                    if approximate:
                        shape.finish(color=(0.95, 0.6, 0.1), fill=(1.0, 0.85, 0.4), fill_opacity=0.18, width=1.2, dashes="[4 3] 0")
                    else:
                        shape.finish(color=None, fill=(1.0, 0.92, 0.2), fill_opacity=0.42, width=0)
                shape.commit()
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
            return pix.tobytes("png")
        finally:
            doc.close()


def render_region(pdf_path: str, page_number: int, bbox: list[float], dpi: int = 150, margin: float = 8.0) -> bytes:
    with _LOCK:
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_number - 1]
            clip = fitz.Rect(bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin) & page.rect
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=clip, alpha=False)
            return pix.tobytes("png")
        finally:
            doc.close()


def page_count(pdf_path: str) -> int:
    with _LOCK:
        doc = fitz.open(pdf_path)
        try:
            return len(doc)
        finally:
            doc.close()
