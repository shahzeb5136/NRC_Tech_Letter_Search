"""Groundedness evaluation: run a question set through the full pipeline and report
how much of what the model said survived verification, and whether it abstained on
questions the corpus cannot answer.

    python scripts/evaluate.py                       # evaluation/questions.yaml, configured provider
    python scripts/evaluate.py --provider openai --limit 4
    python scripts/evaluate.py --no-judge

Writes evaluation/report_<timestamp>.md and .json (git-ignored by default? no - keep them, they are small).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrc_rag.audit.log import AuditLog  # noqa: E402
from nrc_rag.config import get_settings  # noqa: E402
from nrc_rag.index.embeddings import Embedder, try_load_reranker  # noqa: E402
from nrc_rag.index.retriever import HybridRetriever  # noqa: E402
from nrc_rag.index.store import IndexStore  # noqa: E402
from nrc_rag.llm import get_provider  # noqa: E402
from nrc_rag.verify.engine import GroundedEngine  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=str(Path(__file__).resolve().parent.parent / "evaluation" / "questions.yaml"))
    ap.add_argument("--provider", default=None, help="anthropic | openai | google (default: configured)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    s = get_settings()
    provider = get_provider(s, args.provider)
    if provider is None:
        print("No provider configured (set an API key in .env)")
        return 2
    store = IndexStore(s.index_dir)
    retriever = HybridRetriever(store, Embedder(s.embedding_model, s.embedding_max_seq), s, try_load_reranker(s.reranker_model) if s.enable_reranker else None)
    engine = GroundedEngine(store, retriever, s, AuditLog(s.audit_dir), provider)

    qs = yaml.safe_load(Path(args.questions).read_text(encoding="utf-8"))["questions"]
    if args.limit:
        qs = qs[: args.limit]

    results = []
    t0 = time.time()
    for i, item in enumerate(qs, start=1):
        q = item["q"]
        print(f"[{i}/{len(qs)}] {q}")
        va = engine.answer(q, use_judge=not args.no_judge, attach_images=not args.no_images)
        st = va.stats
        cited_docs = sorted({c.check.doc_id for c in va.citations})
        expected = item.get("docs") or []
        hit = (not expected) or any(d in cited_docs for d in expected)
        ok_abstain = (item.get("answerable", True) is False) and va.status == "not_found"
        results.append(
            {
                "question": q,
                "answerable": item.get("answerable", True),
                "status": va.status,
                "abstained_correctly": ok_abstain if item.get("answerable", True) is False else None,
                "expected_docs": expected,
                "cited_docs": cited_docs,
                "expected_doc_cited": hit if item.get("answerable", True) else None,
                "claims_total": st.get("claims_total", 0),
                "claims_displayed": st.get("claims_displayed", 0),
                "claims_withheld": st.get("claims_withheld", 0),
                "quotes_exact": st.get("quotes_exact", 0),
                "quotes_fuzzy": st.get("quotes_fuzzy", 0),
                "quotes_failed": st.get("quotes_failed", 0),
                "seconds": va.timings.get("total_s"),
                "audit_id": va.audit_id,
                "error": va.error,
            }
        )
        print(f"    -> {va.status}: displayed {st.get('claims_displayed', 0)}/{st.get('claims_total', 0)}, quotes exact/fuzzy/failed {st.get('quotes_exact', 0)}/{st.get('quotes_fuzzy', 0)}/{st.get('quotes_failed', 0)}, {va.timings.get('total_s')}s{(' ERROR ' + va.error) if va.error else ''}")

    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    tot_claims = sum(r["claims_total"] for r in answerable)
    tot_disp = sum(r["claims_displayed"] for r in answerable)
    tot_q = sum(r["quotes_exact"] + r["quotes_fuzzy"] + r["quotes_failed"] for r in results)
    tot_exact = sum(r["quotes_exact"] for r in results)
    tot_fuzzy = sum(r["quotes_fuzzy"] for r in results)
    summary = {
        "provider": f"{provider.name}/{provider.model}",
        "questions": len(results),
        "answerable_answered": sum(1 for r in answerable if r["status"] in ("answered", "partial")),
        "answerable_total": len(answerable),
        "expected_doc_cited_rate": (sum(1 for r in answerable if r["expected_doc_cited"]) / len(answerable)) if answerable else None,
        "abstention_rate_on_unanswerable": (sum(1 for r in unanswerable if r["abstained_correctly"]) / len(unanswerable)) if unanswerable else None,
        "claims_displayed_over_generated": (tot_disp / tot_claims) if tot_claims else None,
        "quotes_exact_rate": (tot_exact / tot_q) if tot_q else None,
        "quotes_fuzzy_rate": (tot_fuzzy / tot_q) if tot_q else None,
        "quotes_failed_rate": ((tot_q - tot_exact - tot_fuzzy) / tot_q) if tot_q else None,
        "errors": sum(1 for r in results if r["error"]),
        "elapsed_s": round(time.time() - t0, 1),
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.questions).resolve().parent
    (out_dir / f"report_{ts}.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [f"# Groundedness evaluation — {ts}", "", f"Provider: `{summary['provider']}`", ""]
    md.append("| Metric | Value |\n|---|---|")
    for k, v in summary.items():
        md.append(f"| {k} | {v if not isinstance(v, float) else f'{v:.3f}'} |")
    md.append("")
    md.append("| Question | Answerable | Status | Displayed/Generated | Quotes exact/fuzzy/failed | Expected doc cited | Seconds |\n|---|---|---|---|---|---|---|")
    for r in results:
        md.append(f"| {r['question'][:70]} | {r['answerable']} | {r['status']} | {r['claims_displayed']}/{r['claims_total']} | {r['quotes_exact']}/{r['quotes_fuzzy']}/{r['quotes_failed']} | {r['expected_doc_cited']} | {r['seconds']} |")
    (out_dir / f"report_{ts}.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"report written to {out_dir / f'report_{ts}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
