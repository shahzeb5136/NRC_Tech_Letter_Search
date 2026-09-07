"""Anthropic Claude provider.

Uses the Messages API *citations* feature: every context excerpt is passed as a
``document`` block with citations enabled, so the API itself returns the exact
``cited_text`` spans for each sentence. Those spans are then re-verified locally
against the stored source text like any other quote (belt and braces).
"""

from __future__ import annotations

import logging
import re
from typing import Any

import anthropic

from nrc_rag.config import Settings
from nrc_rag.llm.base import ContextItem, Generation, LLMProvider, b64
from nrc_rag.llm.prompts import (
    FIGURE_DESCRIPTION_SYSTEM,
    GROUNDED_SYSTEM_CITATIONS,
    SUPPORT_JUDGE_SYSTEM,
    figure_description_prompt,
    support_judge_prompt,
)
from nrc_rag.llm.schema import Claim, Evidence, ModelAnswer, SupportVerdict, parse_verdict

log = logging.getLogger(__name__)

_SENTENCE_LIKE = re.compile(r"[A-Za-z]{3,}")


def _is_connective(text: str) -> bool:
    """Uncited fragments that carry no facts (e.g. 'Additionally, ' or 'The report states that')."""
    words = re.findall(r"[A-Za-z0-9%°]+", text)
    if len(words) > 9:
        return False
    if re.search(r"\d", text):
        return False
    return True


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.anthropic_model
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_seconds, max_retries=2)

    # ----------------------------------------------------------------- helpers
    def _create(self, **kwargs: Any):
        """messages.create with optional server-side refusal fallbacks (opt-in)."""
        if self.settings.anthropic_enable_fallbacks:
            extra_headers = dict(kwargs.pop("extra_headers", {}) or {})
            extra_headers["anthropic-beta"] = "server-side-fallback-2026-07-01"
            extra_body = dict(kwargs.pop("extra_body", {}) or {})
            extra_body["fallbacks"] = "default"
            kwargs["extra_headers"] = extra_headers
            kwargs["extra_body"] = extra_body
        # Long answers: stream to avoid HTTP timeouts, then collect the final message.
        with self.client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()

    @staticmethod
    def _check_stop(msg) -> None:
        if msg.stop_reason == "refusal":
            details = getattr(msg, "stop_details", None)
            cat = getattr(details, "category", None) if details else None
            raise RuntimeError(f"model declined the request (refusal{f', category={cat}' if cat else ''})")
        if msg.stop_reason == "max_tokens":
            raise RuntimeError("model output was truncated (max_tokens); retry with a narrower question")

    @staticmethod
    def _usage(msg) -> dict:
        u = getattr(msg, "usage", None)
        if not u:
            return {}
        return {
            "input_tokens": getattr(u, "input_tokens", None),
            "output_tokens": getattr(u, "output_tokens", None),
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None),
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None),
        }

    # ------------------------------------------------------------ generation
    def generate_grounded(self, question: str, items: list[ContextItem]) -> Generation:
        content: list[dict] = []
        for it in items:
            if it.image_png:
                content.append({"type": "text", "text": f"Image attached for figure document {it.chunk_id}:"})
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(it.image_png)}})
            content.append(
                {
                    "type": "document",
                    "source": {"type": "text", "media_type": "text/plain", "data": it.text},
                    "title": it.title,
                    "context": it.header_line(),
                    "citations": {"enabled": True},
                }
            )
        content.append({"type": "text", "text": f"QUESTION\n{question}\n\nAnswer using only the documents above; cite every sentence."})

        msg = self._create(
            model=self.model,
            max_tokens=16000,
            system=[{"type": "text", "text": GROUNDED_SYSTEM_CITATIONS, "cache_control": {"type": "ephemeral"}}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.settings.anthropic_effort},
            messages=[{"role": "user", "content": content}],
        )
        self._check_stop(msg)

        by_index = {i: it for i, it in enumerate(items)}
        claims: list[Claim] = []
        raw_parts: list[str] = []
        full_text_parts: list[str] = []
        for block in msg.content:
            if block.type != "text":
                continue
            text = block.text
            raw_parts.append(text)
            full_text_parts.append(text)
            cits = getattr(block, "citations", None) or []
            evidence: list[Evidence] = []
            for c in cits:
                idx = getattr(c, "document_index", None)
                cited = getattr(c, "cited_text", "") or ""
                it = by_index.get(idx) if idx is not None else None
                if it is not None and cited.strip():
                    evidence.append(Evidence(chunk_id=it.chunk_id, quote=cited.strip()))
            stripped = text.strip()
            if not stripped:
                continue
            if evidence:
                claims.append(Claim(text=stripped, evidence=evidence, kind="statement"))
            else:
                claims.append(Claim(text=stripped, evidence=[], kind="connective" if _is_connective(stripped) else "statement"))

        full_text = "".join(full_text_parts).strip()
        status = "answered"
        if re.fullmatch(r"\W*NOT_FOUND\W*", full_text, re.I) or not any(c.evidence for c in claims):
            status = "not_found"
            claims = [c for c in claims if c.evidence]  # nothing verifiable
        elif re.search(r"not covered by the excerpts", full_text, re.I):
            status = "partial"
        notes = [c.text for c in claims if not c.evidence and re.match(r"\s*not covered by the excerpts", c.text, re.I)]
        claims = [c for c in claims if c.text not in notes]

        answer = ModelAnswer(status=status, claims=claims, notes=notes)
        return Generation(
            answer=answer,
            raw_text="".join(raw_parts),
            provider=self.name,
            model=getattr(msg, "model", self.model) or self.model,
            usage=self._usage(msg),
            request_meta={"effort": self.settings.anthropic_effort, "citations": True, "request_id": getattr(msg, "_request_id", None)},
            mode="citations",
        )

    # ---------------------------------------------------------------- judge
    def judge_support(self, claim_text: str, quotes: list[str]) -> SupportVerdict:
        msg = self._create(
            model=self.model,
            max_tokens=2000,
            system=SUPPORT_JUDGE_SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": support_judge_prompt(claim_text, quotes)}],
        )
        self._check_stop(msg)
        text = "".join(b.text for b in msg.content if b.type == "text")
        return parse_verdict(text)

    # -------------------------------------------------------------- figures
    def describe_figure(self, png: bytes, caption: str, nearby_text: str) -> str:
        msg = self._create(
            model=self.model,
            max_tokens=2000,
            system=FIGURE_DESCRIPTION_SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(png)}},
                        {"type": "text", "text": figure_description_prompt(caption, nearby_text)},
                    ],
                }
            ],
        )
        self._check_stop(msg)
        return "".join(b.text for b in msg.content if b.type == "text").strip()
