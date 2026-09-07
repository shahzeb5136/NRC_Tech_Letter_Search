"""Quote verification.

Given a quote the model attributed to a chunk, prove (or fail to prove) that the
quote exists verbatim in that chunk (or, failing that, on the same page). This is
pure string processing - no model is involved - so the result is reproducible.

Status
------
exact   the normalised quote is a substring of the normalised source
fuzzy   best-window similarity >= threshold (typographic drift, hyphenation, tables)
failed  nothing close enough; the evidence is rejected
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Literal, Optional

from nrc_rag.index.store import ChunkRow, PageRow
from nrc_rag.utils import dehyphenate, normalize_text

QuoteStatus = Literal["exact", "fuzzy", "failed"]

_PUNCT_RE = re.compile(r"[|*_`#>\[\]\(\)\"'“”‘’]+")
_WS_RE = re.compile(r"\s+")


def _strip_markup(s: str) -> str:
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", s)).strip()


def source_variants(text: str) -> list[str]:
    """Normalised renderings of a source text that a faithful quote may match."""
    base = normalize_text(text)
    dehyph = normalize_text(dehyphenate(text))
    out = [base]
    if dehyph != base:
        out.append(dehyph)
    for v in list(out):
        sm = _strip_markup(v)
        if sm != v:
            out.append(sm)
    return out


def quote_variants(quote: str) -> list[str]:
    base = normalize_text(quote)
    out = [base]
    sm = _strip_markup(base)
    if sm != base:
        out.append(sm)
    # models sometimes wrap quotes in quotation marks or trailing ellipses
    trimmed = base.strip("\"'“”‘’ .…")
    if trimmed and trimmed != base:
        out.append(trimmed)
    return [v for v in out if v]


@dataclass
class QuoteCheck:
    chunk_id: str
    quote: str
    status: QuoteStatus
    score: float
    reason: str = ""
    matched_text: str = ""
    location: Literal["chunk", "page", "none"] = "none"
    doc_id: str = ""
    page_number: int = 0
    kind: str = "text"
    section: str = ""
    rects: list[list[float]] = field(default_factory=list)
    approximate_location: bool = False
    in_page_text: bool = False  # the quote also exists verbatim in the page text (true text evidence)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "quote": self.quote,
            "status": self.status,
            "score": round(self.score, 4),
            "reason": self.reason,
            "matched_text": self.matched_text,
            "location": self.location,
            "doc_id": self.doc_id,
            "page": self.page_number,
            "kind": self.kind,
            "section": self.section,
            "rects": [[round(x, 1) for x in r] for r in self.rects],
            "approximate_location": self.approximate_location,
            "in_page_text": self.in_page_text,
        }


def best_window_similarity(source: str, needle: str) -> tuple[float, str]:
    """Best difflib ratio between *needle* and a window of *source* of similar length."""
    if not source or not needle:
        return 0.0, ""
    n = len(needle)
    if len(source) <= n * 1.5:
        return SequenceMatcher(None, source, needle, autojunk=False).ratio(), source
    best_ratio, best_win = 0.0, ""
    # anchor on the longest common substring first (cheap and usually right)
    sm = SequenceMatcher(None, source, needle, autojunk=False)
    m = sm.find_longest_match(0, len(source), 0, n)
    anchors = set()
    if m.size > 0:
        anchors.add(max(0, m.a - m.b))
    # plus a coarse scan
    step = max(1, n // 4)
    for start in range(0, max(1, len(source) - n // 2), step):
        anchors.add(start)
    for a in anchors:
        for extra in (0, n // 6):
            win = source[max(0, a - extra) : a + n + extra]
            if not win:
                continue
            r = SequenceMatcher(None, win, needle, autojunk=False).ratio()
            if r > best_ratio:
                best_ratio, best_win = r, win
    return best_ratio, best_win


def verify_quote(
    quote: str,
    chunk: Optional[ChunkRow],
    page: Optional[PageRow],
    fuzzy_threshold: float = 0.92,
    min_words: int = 3,
    min_chars: int = 12,
) -> QuoteCheck:
    base = QuoteCheck(chunk_id=chunk.chunk_id if chunk else "", quote=quote, status="failed", score=0.0)
    if chunk is None:
        base.reason = "cited chunk id is not part of the retrieved context"
        return base
    base.doc_id = chunk.doc_id
    base.page_number = chunk.page_number
    base.kind = chunk.kind
    base.section = chunk.section

    q_words = re.findall(r"\w+", quote)
    if len(q_words) < min_words or len(quote.strip()) < min_chars:
        base.reason = f"quote too short to verify ({len(q_words)} words)"
        return base

    qvars = quote_variants(quote)
    chunk_vars = source_variants(chunk.text)
    page_vars = source_variants(page.text) if page is not None else []

    base.in_page_text = any(qv in sv for qv in qvars for sv in page_vars)
    for qv in qvars:
        for sv in chunk_vars:
            if qv in sv:
                base.status, base.score, base.location, base.matched_text = "exact", 1.0, "chunk", quote.strip()
                base.reason = "verbatim match in cited chunk"
                return base
    if base.in_page_text:
        base.status, base.score, base.location, base.matched_text = "exact", 1.0, "page", quote.strip()
        base.reason = "verbatim match on the cited page (outside the chunk boundary)"
        return base

    best, best_win, best_loc = 0.0, "", "none"
    for qv in qvars:
        for sv in chunk_vars:
            r, w = best_window_similarity(sv, qv)
            if r > best:
                best, best_win, best_loc = r, w, "chunk"
        for sv in page_vars:
            r, w = best_window_similarity(sv, qv)
            if r > best:
                best, best_win, best_loc = r, w, "page"
    base.score = best
    if best >= fuzzy_threshold:
        base.status = "fuzzy"
        base.location = best_loc  # type: ignore[assignment]
        base.matched_text = best_win
        base.reason = f"near-verbatim match (similarity {best:.2f}); differences are typographic"
    else:
        base.reason = f"quote not found in the cited source (best similarity {best:.2f})"
    return base
