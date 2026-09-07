"""PDF extraction with PyMuPDF.

For every page we keep the text *with block geometry* (so a chunk can be located on
the page later), the outline-derived section, detected tables (as markdown) and
figures (raster images and vector-drawing clusters, rendered to PNG together with
their caption). Everything carries page-level provenance.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from nrc_rag.utils import normalize_text, sha1_bytes, sha256_file, smart_title_case

log = logging.getLogger(__name__)

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
TLR_RE = re.compile(
    r"TLR[^A-Za-z0-9]{0,3}RES[^A-Za-z0-9]{0,3}DE(?:[^A-Za-z0-9]{0,3}(REB))?[^A-Za-z0-9]{0,3}(\d{4})[^A-Za-z0-9]{0,3}(\d{2,3})",
    re.I,
)
DATE_RE = re.compile(rf"\b({MONTHS})\s+(\d{{4}})\b")
CAPTION_RE = re.compile(r"(?m)^[ 	]*(Figure|Fig\.|FIGURE)[ 	]*([A-Z]?\d+(?:[.\-–]\d+)*[a-z]?)(?:[ 	]*[.:\-–][ 	]*|[ 	]+(?=[A-Z(]))(\S.*)?")
TABLE_CAPTION_RE = re.compile(r"^\s*(Table|TABLE)\s*([A-Z]?\d+(?:[.\-–]\d+)*[a-z]?)\s*[.:\-–]?\s*(.*)", re.S)
HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\.?\s+([A-Z][^\n]{2,120})\s*$")
DOTTED_LEADER_RE = re.compile(r"(?:\.\s?){5,}\s*\d+\s*$|\.{5,}")
PAGE_NUMBER_RE = re.compile(r"^\s*(\d{1,4}|[ivxlcIVXLC]{1,7})\s*$")
FRONT_MATTER_RE = re.compile(
    r"^(table of )?contents$|^list of (figures|tables|equations)$|^figures$|^tables$|^equations$|^acronyms|^abbreviations",
    re.I,
)
SKIP_TITLE_LINE_RE = re.compile(
    r"^(technical\s+)?letter\s+report$|^\[?TLR|^ML\d|^SAND\d|^PNNL-|^LA-UR|^INL/|^ANL-|^ORNL|^Date\b|^Choose an item|^Prepared",
    re.I,
)
ORGANIZATIONS = [
    ("Pacific Northwest National Laboratory", "PNNL"),
    ("Argonne National Laboratory", "ANL"),
    ("Idaho National Laboratory", "INL"),
    ("Los Alamos National Laboratory", "LANL"),
    ("Sandia National Laboratories", "SNL"),
    ("Oak Ridge National Laboratory", "ORNL"),
    ("Brookhaven National Laboratory", "BNL"),
    ("Lawrence Livermore National Laboratory", "LLNL"),
]


# --------------------------------------------------------------------------- data


@dataclass
class Block:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    char_start: int
    char_end: int

    @property
    def bbox(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass
class TableData:
    bbox: list[float]
    markdown: str
    rows: int
    cols: int
    caption: str = ""


@dataclass
class FigureData:
    figure_id: str
    doc_id: str
    page_number: int
    bbox: list[float]
    caption: str
    image_path: str
    sha1: str
    source: str  # raster | vector | mixed
    nearby_text: str = ""
    description: str = ""


@dataclass
class PageData:
    doc_id: str
    page_number: int  # 1-based, as shown by PDF viewers
    label: str
    width: float
    height: float
    text: str
    blocks: list[Block]
    section: str = ""
    section_path: str = ""
    is_toc: bool = False
    tables: list[TableData] = field(default_factory=list)
    figures: list[FigureData] = field(default_factory=list)


@dataclass
class DocumentData:
    doc_id: str
    path: str
    sha256: str
    title: str
    tlr_number: str
    report_date: str
    organization: str
    page_count: int
    toc: list[list]  # [level, title, page]
    pages: list[PageData]


# ------------------------------------------------------------------ helpers


def _rect_area(r: fitz.Rect) -> float:
    return max(0.0, r.width) * max(0.0, r.height)


def _overlap_fraction(a: fitz.Rect, b: fitz.Rect) -> float:
    """Fraction of *a* covered by *b*."""
    inter = fitz.Rect(a) & fitz.Rect(b)
    if inter.is_empty or _rect_area(a) == 0:
        return 0.0
    return _rect_area(inter) / _rect_area(a)


def _merge_rects(rects: list[fitz.Rect]) -> list[fitz.Rect]:
    rects = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out: list[fitz.Rect] = []
        while rects:
            r = rects.pop()
            merged = False
            for i, o in enumerate(out):
                if (r & o).is_empty is False or _overlap_fraction(r, o) > 0.3 or _overlap_fraction(o, r) > 0.3:
                    out[i] = o | r
                    merged = True
                    changed = True
                    break
            if not merged:
                out.append(r)
        rects = out
    return rects


def _clean_metadata_title(t: str) -> str:
    t = (t or "").strip().strip('"').strip()
    m = TLR_RE.match(t)
    if m:
        t = t[m.end():]
    t = re.sub(r"^[\s,:\-\]\)\"_]+", "", t)
    t = t.replace("_", " ").strip().strip('"')
    return t


def _title_from_first_page(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title_lines: list[str] = []
    started = False
    for ln in lines[:30]:
        if SKIP_TITLE_LINE_RE.match(ln) or DATE_RE.search(ln):
            if started:
                break
            continue
        title_lines.append(ln)
        started = True
        if len(" ".join(title_lines)) > 220:
            break
    title = ""
    for ln in title_lines:
        if title.endswith("-") and ln[:1].isalpha():
            title += ln  # word hyphenated across a line break, e.g. "High-" + "Temperature"
        else:
            title += (" " if title else "") + ln
    return title.strip()


def _title_is_plausible(t: str) -> bool:
    if len(t) < 15 or len(t.split()) < 4:
        return False
    if re.search(r"national laborator|university|\b[A-Z]\.\s?[A-Z]", t, re.I):
        return False
    if not any(len(w) >= 6 for w in re.findall(r"[A-Za-z]+", t)):
        return False
    return True


def _extract_doc_metadata(doc: fitz.Document, doc_id: str) -> tuple[str, str, str, str]:
    """Return (title, tlr_number, report_date, organization)."""
    first_text = doc[0].get_text("text") if len(doc) else ""
    first_three = "\n".join(doc[i].get_text("text") for i in range(min(3, len(doc))))

    tlr = ""
    m = TLR_RE.search(first_text) or TLR_RE.search(doc.metadata.get("title", "") or "")
    if m:
        tlr = f"TLR-RES/DE/{'REB-' if m.group(1) else ''}{m.group(2)}-{m.group(3)}"

    page_title = _title_from_first_page(first_text)
    meta_title = _clean_metadata_title(doc.metadata.get("title", "") or "")
    if _title_is_plausible(page_title):
        title = page_title
    elif _title_is_plausible(meta_title):
        title = meta_title
    else:
        title = page_title or meta_title or doc_id
    title = title.replace("�", "-")
    if title.isupper():
        title = smart_title_case(title)
    title = re.sub(r"\s+", " ", title).strip(" .")

    date = ""
    dm = DATE_RE.search(first_text)
    if dm:
        date = f"{dm.group(1)} {dm.group(2)}"

    org = ""
    for long_name, short in ORGANIZATIONS:
        if re.search(re.escape(long_name), first_three, re.I) or re.search(rf"\b{short}\b", first_three):
            org = long_name
            break
    return title, tlr, date, org


def _repeated_edge_lines(doc: fitz.Document) -> set[str]:
    """Header/footer lines that repeat across many pages (digits masked)."""
    counts: Counter = Counter()
    n = len(doc)
    for page in doc:
        h = page.rect.height
        seen: set[str] = set()
        for b in page.get_text("blocks", sort=True):
            if len(b) < 7 or b[6] != 0:
                continue
            if not (b[3] < 0.12 * h or b[1] > 0.88 * h):
                continue
            for ln in b[4].splitlines():
                ln = ln.strip()
                if ln:
                    key = re.sub(r"\d+", "#", normalize_text(ln))
                    if 0 < len(key) <= 220:
                        seen.add(key)
        for key in seen:
            counts[key] += 1
    threshold = max(3, int(0.25 * n))
    return {k for k, c in counts.items() if c >= threshold}


def _is_edge_noise(text: str, y0: float, y1: float, page_h: float, repeated: set[str]) -> bool:
    if not (y1 < 0.12 * page_h or y0 > 0.88 * page_h):
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    keys = [re.sub(r"\d+", "#", normalize_text(ln)) for ln in lines]
    noise = sum(1 for ln, k in zip(lines, keys) if PAGE_NUMBER_RE.match(ln) or k in repeated)
    if noise == len(lines) or (len(lines) >= 2 and noise / len(lines) >= 0.6):
        return True
    short = len(lines) <= 3 and sum(len(ln) for ln in lines) <= 80
    return short and noise > 0


def _page_blocks(page: fitz.Page, repeated: set[str]) -> tuple[str, list[Block]]:
    raw = page.get_text("blocks", sort=True)
    blocks: list[Block] = []
    parts: list[str] = []
    pos = 0
    for b in raw:
        if len(b) < 7 or b[6] != 0:
            continue  # image block
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        text = text.replace("\r", "").rstrip()
        if not text.strip():
            continue
        if _is_edge_noise(text, y0, y1, page.rect.height, repeated):
            continue
        if parts:
            pos += 2  # the "\n\n" separator
        start = pos
        parts.append(text)
        pos += len(text)
        blocks.append(Block(x0, y0, x1, y1, text, start, pos))
    return "\n\n".join(parts), blocks


def _is_toc_page(text: str) -> bool:
    lines = text.splitlines()
    hits = sum(1 for ln in lines if DOTTED_LEADER_RE.search(ln))
    return hits >= 3


def _detect_headings(page: fitz.Page) -> list[tuple[float, int, str]]:
    """Heuristic numbered-heading detection for documents without a usable outline."""
    out = []
    d = page.get_text("dict")
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        spans = [s for ln in b.get("lines", []) for s in ln.get("spans", [])]
        if not spans:
            continue
        text = " ".join(s["text"] for s in spans).strip()
        m = HEADING_RE.match(text)
        if not m:
            continue
        bold = any(("bold" in s["font"].lower()) or (s["flags"] & 16) for s in spans)
        size = max(s["size"] for s in spans)
        if bold or size >= 12:
            level = m.group(1).count(".") + 1
            out.append((b["bbox"][1], level, text))
    return out


def _outline_is_usable(toc: list) -> bool:
    numbered = sum(1 for _, title, _ in toc if re.match(r"^\s*\d+(\.\d+)*\.?\s+\S", title))
    return numbered >= 4


class _SectionIndex:
    """Maps (page, y) -> (deepest section title, breadcrumb path)."""

    def __init__(self, doc: fitz.Document, toc: list):
        self.entries: list[tuple[int, float, int, str]] = []  # (page, y, level, title)
        if _outline_is_usable(toc):
            for level, title, page_no in toc:
                title = re.sub(r"\s+", " ", title).strip()
                if not title or page_no < 1 or page_no > len(doc):
                    continue
                if re.search(r"\.pdf$|_|title page|cover", title, re.I) and not re.match(r"^\d", title):
                    continue
                y = 0.0
                try:
                    hits = doc[page_no - 1].search_for(title[:60])
                    if hits:
                        y = min(h.y0 for h in hits)
                except Exception:
                    pass
                self.entries.append((page_no, y, int(level), title))
        else:
            for i, page in enumerate(doc):
                for y, level, title in _detect_headings(page):
                    self.entries.append((i + 1, y, level, title))
        self.entries.sort(key=lambda e: (e[0], e[1]))
        # breadcrumb paths
        self.paths: list[str] = []
        stack: list[tuple[int, str]] = []
        for _, _, level, title in self.entries:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            self.paths.append(" > ".join(t for _, t in stack))

    def lookup(self, page_no: int, y: float) -> tuple[str, str]:
        best = -1
        for i, (p, ey, _, _) in enumerate(self.entries):
            if p < page_no or (p == page_no and ey <= y + 2.0):
                best = i
            elif p > page_no:
                break
        if best < 0:
            return "", ""
        return self.entries[best][3], self.paths[best]


def _extract_tables(page: fitz.Page, blocks: list[Block]) -> list[TableData]:
    tables: list[TableData] = []
    try:
        found = page.find_tables()
    except Exception as exc:  # pragma: no cover - PyMuPDF edge cases
        log.debug("find_tables failed on page %s: %s", page.number, exc)
        return tables
    for t in found.tables:
        if t.row_count < 2 or t.col_count < 2:
            continue
        try:
            md = t.to_markdown()
        except Exception:
            continue
        if len(md.strip()) < 20:
            continue
        try:
            rows_ = t.extract()
        except Exception:
            rows_ = []
        cells = [c for row in rows_ for c in row]
        non_empty = [str(c).strip() for c in cells if c is not None and str(c).strip()]
        if len(cells) < 4 or len(non_empty) < 0.5 * len(cells):
            continue
        ncols = max((len(r) for r in rows_), default=0)
        filled_cols = sum(
            1 for j in range(ncols)
            if sum(1 for r in rows_ if j < len(r) and r[j] is not None and str(r[j]).strip()) >= 2
        )
        filled_rows = sum(1 for r in rows_ if sum(1 for c in r if c is not None and str(c).strip()) >= 2)
        if filled_cols < 2 or filled_rows < 2:
            continue  # a paragraph or list mis-detected as a table
        if sum(len(c) for c in non_empty) / max(1, len(non_empty)) > 300:
            continue  # title blocks / paragraphs mis-detected as a table
        if _rect_area(fitz.Rect(t.bbox)) > 0.6 * _rect_area(page.rect):
            continue
        bbox = list(t.bbox)
        caption = ""
        for b in blocks:
            if TABLE_CAPTION_RE.match(b.text) and 0 <= bbox[1] - b.y1 <= 90:
                caption = re.sub(r"\s+", " ", b.text).strip()
                break
        tables.append(TableData(bbox=bbox, markdown=md.strip(), rows=t.row_count, cols=t.col_count, caption=caption))
    return tables


def _extract_figures(
    doc: fitz.Document,
    page: fitz.Page,
    doc_id: str,
    blocks: list[Block],
    tables: list[TableData],
    xref_page_counts: Counter,
    figures_dir: Path,
    dpi: int,
) -> list[FigureData]:
    page_no = page.number + 1
    if page_no == 1:
        return []  # cover pages carry logos / backgrounds, not technical figures
    page_area = _rect_area(page.rect)
    candidates: list[tuple[fitz.Rect, str]] = []

    # raster images
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        infos = []
    for info in infos:
        xref = info.get("xref", 0)
        if xref and xref_page_counts.get(xref, 0) > 3:
            continue  # logos and repeated decorations
        r = fitz.Rect(info["bbox"]) & page.rect
        if r.is_empty or _rect_area(r) < 0.015 * page_area or min(r.width, r.height) < 60:
            continue
        candidates.append((r, "raster"))

    # vector drawings
    try:
        clusters = page.cluster_drawings()
    except Exception:
        clusters = []
    table_rects = [fitz.Rect(t.bbox) for t in tables]
    for r in clusters:
        r = fitz.Rect(r) & page.rect
        area = _rect_area(r)
        if r.is_empty or area < 0.04 * page_area or area > 0.85 * page_area or min(r.width, r.height) < 100:
            continue
        if any(_overlap_fraction(r, tr) > 0.4 for tr in table_rects):
            continue
        candidates.append((r, "vector"))

    if not candidates:
        return []

    merged = _merge_rects([c[0] for c in candidates])
    caption_blocks = [b for b in blocks if len(b.text) <= 700 and CAPTION_RE.search(b.text)]

    def _caption_for(rect: fitz.Rect) -> tuple[str, Optional[fitz.Rect]]:
        caption, caption_rect, best = "", None, 1e9
        for b in caption_blocks:
            if min(rect.x1, b.x1) - max(rect.x0, b.x0) <= 0:
                continue
            if b.y1 > rect.y1 and b.y0 >= rect.y1 - 45 and b.y0 - rect.y1 <= 120:
                dist = max(0.0, b.y0 - rect.y1)
            elif b.y0 < rect.y0 and b.y1 <= rect.y0 + 45 and rect.y0 - b.y1 <= 60:
                dist = max(0.0, rect.y0 - b.y1) + 30  # prefer captions below the figure
            else:
                continue
            if dist < best:
                m = CAPTION_RE.search(b.text)
                best = dist
                caption = re.sub(r"\s+", " ", b.text[m.start():]).strip() if m else ""
                caption_rect = fitz.Rect(b.x0, b.y0, b.x1, b.y1)
        return caption, caption_rect

    # assign captions, then merge sub-figures that share one caption block
    assigned: list[tuple[fitz.Rect, str, Optional[fitz.Rect]]] = []
    for rect in merged:
        if _rect_area(rect) > 0.9 * page_area:
            continue
        cap, cap_rect = _caption_for(rect)
        sources = {src for r, src in candidates if _overlap_fraction(r, rect) > 0.5}
        if not cap and sources == {"vector"}:
            continue  # uncaptioned vector clusters are usually decorations, equations or rules
        assigned.append((rect, cap, cap_rect))
    grouped: list[tuple[fitz.Rect, str, Optional[fitz.Rect]]] = []
    for rect, cap, cap_rect in assigned:
        for i, (g_rect, g_cap, g_cap_rect) in enumerate(grouped):
            if cap_rect is not None and g_cap_rect == cap_rect:
                grouped[i] = (g_rect | rect, g_cap, g_cap_rect)
                break
        else:
            grouped.append((rect, cap, cap_rect))

    # multi-panel figures: attach uncaptioned panels to an adjacent captioned group
    changed = True
    while changed:
        changed = False
        for i, (rect, cap, cap_rect) in enumerate(grouped):
            if cap:
                continue
            for j, (g_rect, g_cap, g_cap_rect) in enumerate(grouped):
                if i == j or not g_cap:
                    continue
                h_overlap = min(rect.x1, g_rect.x1) - max(rect.x0, g_rect.x0)
                v_gap = max(rect.y0, g_rect.y0) - min(rect.y1, g_rect.y1)
                v_overlap = min(rect.y1, g_rect.y1) - max(rect.y0, g_rect.y0)
                h_gap = max(rect.x0, g_rect.x0) - min(rect.x1, g_rect.x1)
                if (h_overlap > 0 and v_gap <= 25) or (v_overlap > 0 and h_gap <= 40):
                    grouped[j] = (g_rect | rect, g_cap, g_cap_rect)
                    grouped.pop(i)
                    changed = True
                    break
            if changed:
                break

    figures: list[FigureData] = []
    for n, (rect, caption, caption_rect) in enumerate(sorted(grouped, key=lambda g: (g[0].y0, g[0].x0)), start=1):
        sources = {src for r, src in candidates if _overlap_fraction(r, rect) > 0.5}
        source = "mixed" if len(sources) > 1 else (next(iter(sources)) if sources else "vector")

        nearby = []
        for b in blocks:
            if caption_rect is not None and fitz.Rect(b.x0, b.y0, b.x1, b.y1) == caption_rect:
                continue
            if b.y1 <= rect.y0 and rect.y0 - b.y1 <= 150:
                nearby.append(b.text)
        nearby_text = re.sub(r"\s+", " ", " ".join(nearby)).strip()[-400:]

        clip = fitz.Rect(rect)
        if caption_rect is not None:
            clip = clip | caption_rect
        clip = fitz.Rect(clip.x0 - 6, clip.y0 - 6, clip.x1 + 6, clip.y1 + 6) & page.rect
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=clip, alpha=False)
            png = pix.tobytes("png")
        except Exception as exc:
            log.warning("figure render failed %s p%s: %s", doc_id, page_no, exc)
            continue
        figure_id = f"{doc_id}:p{page_no}:f{n}"
        out_dir = figures_dir / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"p{page_no:03d}_f{n}.png"
        out_path.write_bytes(png)
        figures.append(
            FigureData(
                figure_id=figure_id,
                doc_id=doc_id,
                page_number=page_no,
                bbox=[rect.x0, rect.y0, rect.x1, rect.y1],
                caption=caption,
                image_path=str(out_path),
                sha1=sha1_bytes(png),
                source=source,
                nearby_text=nearby_text,
            )
        )
    return figures


# ------------------------------------------------------------------ public API


def doc_id_from_path(path: Path) -> str:
    return path.stem


def extract_document(pdf_path: Path, figures_dir: Path, figure_dpi: int = 150, extract_figures: bool = True) -> DocumentData:
    pdf_path = Path(pdf_path)
    doc_id = doc_id_from_path(pdf_path)
    doc = fitz.open(pdf_path)
    try:
        title, tlr, date, org = _extract_doc_metadata(doc, doc_id)
        toc = [[int(lvl), str(t), int(p)] for lvl, t, p in doc.get_toc()]
        sections = _SectionIndex(doc, toc)
        repeated = _repeated_edge_lines(doc)

        xref_counts: Counter = Counter()
        for page in doc:
            for img in page.get_images(full=True):
                xref_counts[img[0]] += 1

        pages: list[PageData] = []
        for page in doc:
            page_no = page.number + 1
            text, blocks = _page_blocks(page, repeated)
            is_toc = _is_toc_page(text)
            section, section_path = sections.lookup(page_no, 0.0)
            if FRONT_MATTER_RE.match(section or ""):
                is_toc = True
            tables = [] if is_toc else _extract_tables(page, blocks)
            figures = []
            if extract_figures and not is_toc:
                figures = _extract_figures(doc, page, doc_id, blocks, tables, xref_counts, figures_dir, figure_dpi)
            pages.append(
                PageData(
                    doc_id=doc_id,
                    page_number=page_no,
                    label=page.get_label() or "",
                    width=page.rect.width,
                    height=page.rect.height,
                    text=text,
                    blocks=blocks,
                    section=section,
                    section_path=section_path,
                    is_toc=is_toc,
                    tables=tables,
                    figures=figures,
                )
            )
        # attach fine-grained sections to blocks via the page lookup (used by the chunker)
        for p in pages:
            p._section_lookup = lambda y, _p=p.page_number, _s=sections: _s.lookup(_p, y)  # type: ignore[attr-defined]

        return DocumentData(
            doc_id=doc_id,
            path=str(pdf_path),
            sha256=sha256_file(pdf_path),
            title=title,
            tlr_number=tlr,
            report_date=date,
            organization=org,
            page_count=len(doc),
            toc=toc,
            pages=pages,
        )
    finally:
        doc.close()
