"""Provider-independent interfaces and context assembly."""

from __future__ import annotations

import base64
import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from nrc_rag.index.retriever import RetrievedChunk
from nrc_rag.llm.schema import ModelAnswer, SupportVerdict

log = logging.getLogger(__name__)

MAX_IMAGE_EDGE = 1400


@dataclass
class ContextItem:
    """One excerpt as presented to the model."""

    chunk_id: str
    doc_id: str
    page_number: int
    kind: str
    section: str
    doc_title: str
    tlr_number: str
    text: str
    image_png: Optional[bytes] = None  # for figure chunks, when attached
    figure_id: Optional[str] = None

    @property
    def title(self) -> str:
        t = f"{self.chunk_id} | {self.tlr_number or self.doc_id} | page {self.page_number}"
        if self.section:
            t += f" | {self.section[:80]}"
        return t

    def header_line(self) -> str:
        head = f"[chunk_id: {self.chunk_id}] document: {self.doc_id}"
        if self.tlr_number:
            head += f" ({self.tlr_number})"
        head += f", page {self.page_number}"
        if self.section:
            head += f", section: {self.section[:100]}"
        if self.kind != "text":
            head += f", type: {self.kind}"
        return head


@dataclass
class Generation:
    """What came back from a provider, before verification."""

    answer: ModelAnswer
    raw_text: str
    provider: str
    model: str
    usage: dict = field(default_factory=dict)
    request_meta: dict = field(default_factory=dict)
    mode: str = "json"  # json | citations


def _downscale_png(png: bytes, max_edge: int = MAX_IMAGE_EDGE) -> bytes:
    try:
        from PIL import Image

        im = Image.open(io.BytesIO(png))
        w, h = im.size
        if max(w, h) <= max_edge:
            return png
        scale = max_edge / float(max(w, h))
        im = im.convert("RGB").resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception as exc:  # pragma: no cover
        log.warning("image downscale failed: %s", exc)
        return png


def build_context(retrieved: list[RetrievedChunk], figure_image, max_figures: int, attach_images: bool = True) -> list[ContextItem]:
    """Turn retrieved chunks into context items; attach PNGs for the first *max_figures* figure chunks.

    ``figure_image`` takes a figure id and returns PNG bytes (or None) - it resolves
    a cached file when there is one and re-renders from the PDF when there is not.
    """
    items: list[ContextItem] = []
    attached = 0
    for r in retrieved:
        c = r.chunk
        png = None
        if c.kind == "figure" and attach_images and attached < max_figures and c.figure_id:
            try:
                raw = figure_image(c.figure_id)
                if raw:
                    png = _downscale_png(raw)
                    attached += 1
            except Exception as exc:  # pragma: no cover
                log.warning("could not load figure %s: %s", c.figure_id, exc)
        items.append(
            ContextItem(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                page_number=c.page_number,
                kind=c.kind,
                section=c.section,
                doc_title=r.doc_title,
                tlr_number=r.tlr_number,
                text=c.text,
                image_png=png,
                figure_id=c.figure_id,
            )
        )
    return items


def context_block_text(items: list[ContextItem]) -> str:
    parts = []
    for it in items:
        parts.append(f"{it.header_line()}\n{it.text}\n[end of {it.chunk_id}]")
    return "\n\n".join(parts)


def b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


class LLMProvider(ABC):
    name: str = "base"
    model: str = ""
    supports_vision: bool = True

    @abstractmethod
    def generate_grounded(self, question: str, items: list[ContextItem]) -> Generation:
        """Produce claims + verbatim quotes over the given context items."""

    @abstractmethod
    def judge_support(self, claim_text: str, quotes: list[str]) -> SupportVerdict:
        """Independent check that the quotes entail the claim."""

    @abstractmethod
    def describe_figure(self, png: bytes, caption: str, nearby_text: str) -> str:
        """Describe a figure image for indexing."""

    def describe(self) -> str:
        return f"{self.name} / {self.model}"
