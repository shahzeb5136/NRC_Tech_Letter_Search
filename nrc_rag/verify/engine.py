"""The grounded-answer engine.

    question -> hybrid retrieval -> model claims with quotes -> deterministic quote
    verification -> optional independent support check -> highlight location ->
    audit record

Only claims whose quotes were verified against the stored source text are
displayed as the answer. Everything else is listed separately as withheld.
"""

from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from nrc_rag.audit.log import AuditLog
from nrc_rag.config import APP_VERSION, PIPELINE_VERSION, Settings
from nrc_rag.index.retriever import HybridRetriever, RetrievedChunk
from nrc_rag.index.store import ChunkRow, IndexStore
from nrc_rag.llm.base import Generation, LLMProvider, build_context
from nrc_rag.llm.prompts import PROMPT_VERSION
from nrc_rag.llm.schema import Claim
from nrc_rag.render.page_render import locate_text, quote_candidates
from nrc_rag.utils import normalize_text, utc_now_iso
from nrc_rag.verify.quote_verifier import QuoteCheck, verify_quote

log = logging.getLogger(__name__)

STATUS_LABELS = {
    "verified": "Verified",
    "approximate": "Verified (near-verbatim)",
    "figure": "Figure-derived",
    "connective": "Connective text",
    "withheld": "Withheld",
    "unsupported": "Withheld by support check",
}


@dataclass
class Citation:
    number: int
    check: QuoteCheck
    doc_title: str = ""
    tlr_number: str = ""
    pdf_path: str = ""
    figure_id: Optional[str] = None
    image_path: Optional[str] = None
    caption: str = ""
    page_label: str = ""

    def to_dict(self) -> dict:
        d = self.check.to_dict()
        d.update({"number": self.number, "doc_title": self.doc_title, "tlr_number": self.tlr_number, "pdf_path": self.pdf_path, "figure_id": self.figure_id, "image_path": self.image_path, "caption": self.caption})
        return d


@dataclass
class VerifiedClaim:
    text: str
    kind: str
    status: str
    evidence: list[QuoteCheck] = field(default_factory=list)
    removed: list[QuoteCheck] = field(default_factory=list)
    judge: Optional[dict] = None
    citation_numbers: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def displayed(self) -> bool:
        return self.status in ("verified", "approximate", "figure", "connective")

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "kind": self.kind,
            "status": self.status,
            "evidence": [e.to_dict() for e in self.evidence],
            "removed": [e.to_dict() for e in self.removed],
            "judge": self.judge,
            "citations": self.citation_numbers,
            "notes": self.notes,
        }


@dataclass
class VerifiedAnswer:
    audit_id: str
    question: str
    status: str  # answered | partial | not_found | error
    claims: list[VerifiedClaim] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    provider: str = ""
    model: str = ""
    mode: str = ""
    retrieval: list[dict] = field(default_factory=list)
    context_ids: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    error: str = ""
    raw_output: str = ""
    judge_enabled: bool = False
    usage: dict = field(default_factory=dict)
    created_at: str = ""
    filters: dict = field(default_factory=dict)

    @property
    def displayed_claims(self) -> list[VerifiedClaim]:
        return [c for c in self.claims if c.displayed]

    @property
    def withheld_claims(self) -> list[VerifiedClaim]:
        return [c for c in self.claims if not c.displayed]

    def to_record(self) -> dict:
        return {
            "audit_id": self.audit_id,
            "created_at": self.created_at,
            "question": self.question,
            "status": self.status,
            "filters": self.filters,
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "judge_enabled": self.judge_enabled,
            "usage": self.usage,
            "versions": {"app": APP_VERSION, "pipeline": PIPELINE_VERSION, "prompts": PROMPT_VERSION},
            "retrieval": self.retrieval,
            "context_ids": self.context_ids,
            "raw_model_output": self.raw_output,
            "claims": [c.to_dict() for c in self.claims],
            "citations": [c.to_dict() for c in self.citations],
            "notes": self.notes,
            "warnings": self.warnings,
            "stats": self.stats,
            "timings": self.timings,
            "error": self.error,
        }

    def to_markdown(self) -> str:
        lines = [f"# {self.question}", ""]
        lines.append(f"*Status: {self.status} · provider: {self.provider} / {self.model} · audit id: {self.audit_id} · {self.created_at}*")
        lines.append("")
        if self.status == "not_found":
            lines.append("**The indexed documents do not contain a verifiable answer to this question.**")
        for c in self.displayed_claims:
            marks = "".join(f"[{n}]" for n in c.citation_numbers)
            tag = "" if c.status == "verified" else f" _({STATUS_LABELS.get(c.status, c.status)})_"
            lines.append(f"- {c.text} {marks}{tag}")
        if self.notes:
            lines.append("")
            lines.append("**Notes from the model (not verified):**")
            lines.extend(f"- {n}" for n in self.notes)
        if self.withheld_claims:
            lines.append("")
            lines.append("**Withheld statements (failed verification, not part of the answer):**")
            for c in self.withheld_claims:
                reason = c.judge.get("reason") if c.judge and c.status == "unsupported" else (c.removed[0].reason if c.removed else "no verifiable evidence")
                lines.append(f"- ~~{c.text}~~ — {reason}")
        lines.append("")
        lines.append("## Citations")
        for cit in self.citations:
            ch = cit.check
            loc = f"{cit.tlr_number or ch.doc_id} ({ch.doc_id}), page {ch.page_number}"
            if ch.section:
                loc += f", §{ch.section}"
            lines.append(f"[{cit.number}] {loc} — {ch.status} — “{ch.quote}”")
        lines.append("")
        s = self.stats
        lines.append(f"_Verification: {s.get('claims_displayed', 0)} statements displayed, {s.get('claims_withheld', 0)} withheld; "
                     f"{s.get('quotes_exact', 0)} exact quotes, {s.get('quotes_fuzzy', 0)} near-verbatim, {s.get('quotes_failed', 0)} rejected._")
        return "\n".join(lines)


class GroundedEngine:
    def __init__(self, store: IndexStore, retriever: HybridRetriever, settings: Settings, audit: AuditLog, provider: Optional[LLMProvider] = None) -> None:
        self.store = store
        self.retriever = retriever
        self.settings = settings
        self.audit = audit
        self.provider = provider
        self._doc_meta = {d.doc_id: d for d in store.list_documents()}

    def refresh_metadata(self) -> None:
        self._doc_meta = {d.doc_id: d for d in self.store.list_documents()}

    # --------------------------------------------------------------- search
    def search_only(self, question: str, top_k: Optional[int] = None, doc_ids: Optional[list[str]] = None, kinds: Optional[list[str]] = None) -> list[RetrievedChunk]:
        return self.retriever.search(question, top_k=top_k, doc_ids=doc_ids, kinds=kinds)

    # --------------------------------------------------------------- answer
    def answer(
        self,
        question: str,
        doc_ids: Optional[list[str]] = None,
        top_k: Optional[int] = None,
        use_judge: Optional[bool] = None,
        attach_images: bool = True,
        progress=None,
    ) -> VerifiedAnswer:
        say = progress or (lambda msg: None)
        use_judge = self.settings.enable_support_check if use_judge is None else use_judge
        va = VerifiedAnswer(
            audit_id=self.audit.new_id(),
            question=question.strip(),
            status="error",
            judge_enabled=bool(use_judge),
            created_at=utc_now_iso(),
            filters={"doc_ids": doc_ids or [], "top_k": top_k or self.settings.top_k_final, "attach_images": attach_images},
        )
        timings: dict[str, float] = {}
        t_start = time.time()
        try:
            if self.provider is None:
                raise RuntimeError("No LLM provider is configured. Add an API key to .env (ANTHROPIC_API_KEY, OPENAI_API_KEY or GOOGLE_API_KEY).")
            va.provider, va.model = self.provider.name, self.provider.model

            say("Retrieving passages (dense + lexical, fused and re-ranked)…")
            t = time.time()
            retrieved = self.retriever.search(question, top_k=top_k, doc_ids=doc_ids)
            timings["retrieval_s"] = round(time.time() - t, 3)
            va.retrieval = [r.to_audit() for r in retrieved]
            va.context_ids = [r.chunk.chunk_id for r in retrieved]
            if not retrieved:
                va.status = "not_found"
                va.warnings.append("No passages were retrieved for this question.")
                self._finish(va, timings, t_start)
                return va

            items = build_context(retrieved, self.store.get_figure, self.settings.max_figures_in_context, attach_images)
            context_map: dict[str, ChunkRow] = {r.chunk.chunk_id: r.chunk for r in retrieved}

            say(f"Generating grounded answer with {self.provider.describe()}…")
            t = time.time()
            gen: Generation = self.provider.generate_grounded(question, items)
            timings["generation_s"] = round(time.time() - t, 3)
            va.raw_output = gen.raw_text
            va.usage = gen.usage
            va.mode = gen.mode
            va.model = gen.model or va.model
            va.notes = list(gen.answer.notes)

            say("Verifying every quote against the source text…")
            t = time.time()
            claims = [self._verify_claim(c, context_map) for c in gen.answer.claims]
            timings["verification_s"] = round(time.time() - t, 3)

            if use_judge:
                say("Running independent support check on verified statements…")
                t = time.time()
                to_check = [vc for vc in claims if vc.status in ("verified", "approximate") and vc.evidence]
                if to_check:
                    with ThreadPoolExecutor(max_workers=min(4, len(to_check))) as pool:
                        list(pool.map(self._judge, to_check))
                timings["support_check_s"] = round(time.time() - t, 3)

            say("Locating citations on the source pages…")
            t = time.time()
            va.claims = claims
            va.citations = self._number_citations(claims)
            timings["locate_s"] = round(time.time() - t, 3)

            displayed = [c for c in claims if c.displayed and c.kind == "statement"]
            if gen.answer.status == "not_found" or not displayed:
                va.status = "not_found"
            elif gen.answer.status == "partial" or any(not c.displayed for c in claims):
                va.status = "partial"
            else:
                va.status = "answered"
            if any(c.status in ("withheld", "unsupported") for c in claims):
                va.warnings.append("Some model statements failed verification and were withheld; see the withheld list.")
        except Exception as exc:
            log.error("answer failed: %s\n%s", exc, traceback.format_exc())
            va.status = "error"
            va.error = f"{type(exc).__name__}: {exc}"
        self._finish(va, timings, t_start)
        return va

    # ------------------------------------------------------------ internals
    def _verify_claim(self, claim: Claim, context_map: dict[str, ChunkRow]) -> VerifiedClaim:
        vc = VerifiedClaim(text=claim.text.strip(), kind=claim.kind, status="withheld")
        if claim.kind == "connective" and not claim.evidence:
            vc.status = "connective"
            return vc
        seen: set[tuple[str, str]] = set()
        for ev in claim.evidence:
            key = (ev.chunk_id, normalize_text(ev.quote))
            if key in seen:
                continue
            seen.add(key)
            chunk = context_map.get(ev.chunk_id)
            page = self.store.get_page(chunk.doc_id, chunk.page_number) if chunk else None
            check = verify_quote(
                ev.quote, chunk, page,
                fuzzy_threshold=self.settings.fuzzy_threshold,
                min_words=self.settings.min_quote_words,
                min_chars=self.settings.min_quote_chars,
            )
            if check.status == "failed":
                vc.removed.append(check)
            else:
                vc.evidence.append(check)
        if not vc.evidence:
            vc.status = "withheld"
            if not claim.evidence:
                vc.notes.append("The model gave no evidence for this statement.")
            return vc
        # A statement is "figure-derived" only when its evidence is a figure passage AND the quote does not
        # exist in the page text (i.e. it comes from the AI description of the image). Captions and labels
        # printed on the page are verbatim text evidence, even when the model says "From Figure ...".
        figure_based = any(e.kind == "figure" and not e.in_page_text for e in vc.evidence)
        if figure_based:
            vc.status = "figure"
        elif any(e.status == "fuzzy" for e in vc.evidence):
            vc.status = "approximate"
        else:
            vc.status = "verified"
        if vc.removed:
            vc.notes.append(f"{len(vc.removed)} cited quote(s) could not be found in the source and were removed.")
        return vc

    def _judge(self, vc: VerifiedClaim) -> None:
        quotes = [e.matched_text or e.quote for e in vc.evidence]
        try:
            verdict = self.provider.judge_support(vc.text, quotes)
        except Exception as exc:
            vc.judge = {"verdict": "ERROR", "reason": f"support check failed: {exc}"}
            vc.notes.append("Independent support check could not be completed for this statement.")
            return
        vc.judge = {"verdict": verdict.verdict, "reason": verdict.reason}
        if verdict.verdict == "NOT_SUPPORTED":
            vc.status = "unsupported"
        elif verdict.verdict == "PARTIALLY_SUPPORTED":
            if vc.status == "verified":
                vc.status = "approximate"
            vc.notes.append(f"Support check: partially supported — {verdict.reason}")

    def _number_citations(self, claims: list[VerifiedClaim]) -> list[Citation]:
        citations: list[Citation] = []
        index: dict[tuple[str, str], int] = {}
        for vc in claims:
            if not vc.displayed:
                continue
            vc.citation_numbers = []
            for e in vc.evidence:
                key = (e.chunk_id, normalize_text(e.quote))
                if key not in index:
                    index[key] = len(citations) + 1
                    citations.append(self._make_citation(index[key], e))
                if index[key] not in vc.citation_numbers:
                    vc.citation_numbers.append(index[key])
        return citations

    def _make_citation(self, number: int, check: QuoteCheck) -> Citation:
        doc = self._doc_meta.get(check.doc_id)
        chunk = self.store.get_chunk(check.chunk_id)
        cit = Citation(number=number, check=check, doc_title=doc.title if doc else "", tlr_number=doc.tlr_number if doc else "", pdf_path=doc.path if doc else "")
        bbox = chunk.bbox if chunk else None
        if chunk is not None and chunk.kind == "figure" and chunk.figure_id:
            fig = self.store.get_figure(chunk.figure_id)
            if fig:
                cit.figure_id, cit.image_path, cit.caption = fig.figure_id, fig.image_path, fig.caption
            check.rects, check.approximate_location = ([bbox] if bbox else []), False
        elif chunk is not None and chunk.kind == "table":
            check.rects, check.approximate_location = ([bbox] if bbox else []), False
        elif cit.pdf_path:
            rects, approx = locate_text(cit.pdf_path, check.page_number, quote_candidates(check.quote, check.matched_text), bbox)
            check.rects, check.approximate_location = rects, approx
        return cit

    def _finish(self, va: VerifiedAnswer, timings: dict, t_start: float) -> None:
        timings["total_s"] = round(time.time() - t_start, 3)
        va.timings = timings
        claims = va.claims
        va.stats = {
            "claims_total": len([c for c in claims if c.kind == "statement"]),
            "claims_displayed": len([c for c in claims if c.displayed and c.kind == "statement"]),
            "claims_verified": len([c for c in claims if c.status == "verified"]),
            "claims_approximate": len([c for c in claims if c.status == "approximate"]),
            "claims_figure": len([c for c in claims if c.status == "figure"]),
            "claims_withheld": len([c for c in claims if c.status in ("withheld", "unsupported")]),
            "claims_unsupported": len([c for c in claims if c.status == "unsupported"]),
            "connective_fragments": len([c for c in claims if c.status == "connective"]),
            "quotes_exact": sum(1 for c in claims for e in c.evidence if e.status == "exact"),
            "quotes_fuzzy": sum(1 for c in claims for e in c.evidence if e.status == "fuzzy"),
            "quotes_failed": sum(len(c.removed) for c in claims),
            "citations": len(va.citations),
        }
        try:
            self.audit.append(va.to_record())
        except Exception as exc:  # pragma: no cover
            log.error("audit append failed: %s", exc)
            va.warnings.append(f"Audit record could not be written: {exc}")
