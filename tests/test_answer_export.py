"""Offline tests for the VerifiedAnswer export/record paths (no index, no model)."""

from __future__ import annotations

import json

from nrc_rag.verify.engine import Citation, VerifiedAnswer, VerifiedClaim
from nrc_rag.verify.quote_verifier import QuoteCheck


def _answer() -> VerifiedAnswer:
    ok = QuoteCheck(chunk_id="D1:p3:c1", quote="removes xenon and krypton", status="exact", score=1.0, doc_id="D1", page_number=3, kind="text", section="1.5", rects=[[72, 70, 300, 84]])
    bad = QuoteCheck(chunk_id="D1:p3:c1", quote="operates at 950 degrees for ten years", status="failed", score=0.41, reason="quote not found in the cited source (best similarity 0.41)", doc_id="D1", page_number=3)
    va = VerifiedAnswer(audit_id="abc123", question="What does the off-gas system remove?", status="partial", provider="google", model="gemini-2.5-pro", mode="json", created_at="2026-09-06T00:00:00+00:00")
    va.claims = [
        VerifiedClaim(text="The off-gas system removes xenon and krypton.", kind="statement", status="verified", evidence=[ok], citation_numbers=[1], judge={"verdict": "SUPPORTED", "reason": "stated verbatim"}),
        VerifiedClaim(text="The system operates at 950 degrees for ten years.", kind="statement", status="withheld", removed=[bad]),
    ]
    va.citations = [Citation(number=1, check=ok, doc_title="Doc 1", tlr_number="TLR-1", pdf_path="x.pdf")]
    va.stats = {"claims_total": 2, "claims_displayed": 1, "claims_withheld": 1, "quotes_exact": 1, "quotes_fuzzy": 0, "quotes_failed": 1}
    return va


def test_markdown_export_separates_answer_from_withheld():
    md = _answer().to_markdown()
    assert "removes xenon and krypton" in md
    assert "[1] TLR-1 (D1), page 3, §1.5 — exact" in md
    assert "Withheld statements" in md and "~~The system operates at 950 degrees for ten years.~~" in md
    assert "quote not found" in md


def test_record_is_json_serialisable_and_complete():
    rec = _answer().to_record()
    s = json.dumps(rec)
    assert '"audit_id": "abc123"' in s
    assert rec["claims"][1]["status"] == "withheld" and rec["claims"][1]["removed"][0]["status"] == "failed"
    assert rec["citations"][0]["number"] == 1 and rec["citations"][0]["rects"] == [[72, 70, 300, 84]]
    assert set(rec["versions"]) == {"app", "pipeline", "prompts"}


def test_displayed_vs_withheld_partition():
    va = _answer()
    assert [c.status for c in va.displayed_claims] == ["verified"]
    assert [c.status for c in va.withheld_claims] == ["withheld"]
