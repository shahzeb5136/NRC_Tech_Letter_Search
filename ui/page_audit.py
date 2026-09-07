"""Audit trail: every question, what the model saw, what it said, and what was verified."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ui import common
from ui.theme import badge, esc, hero, inject_css, pill


def render() -> None:
    inject_css()
    st.markdown(hero(common.logo_b64(), "Audit Trail", "Append-only record of every question: retrieval, model output, quote checks, and what was displayed."), unsafe_allow_html=True)
    log = common.audit()
    records = log.read_recent(300)
    if not records:
        st.info("No questions have been asked yet.")
        st.stop()

    rows = []
    for r in records:
        s = r.get("stats", {})
        rows.append(
            {
                "Time (UTC)": r.get("created_at", ""),
                "Question": (r.get("question", "") or "")[:110],
                "Status": r.get("status", ""),
                "Model": f"{r.get('provider', '')}/{r.get('model', '')}",
                "Displayed": s.get("claims_displayed", 0),
                "Withheld": s.get("claims_withheld", 0),
                "Quotes exact/fuzzy/failed": f"{s.get('quotes_exact', 0)}/{s.get('quotes_fuzzy', 0)}/{s.get('quotes_failed', 0)}",
                "Seconds": r.get("timings", {}).get("total_s", ""),
                "Audit id": r.get("audit_id", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=min(500, 40 + 36 * len(rows)))

    total = log.count()
    c1, c2 = st.columns([1, 5])
    with c1:
        st.download_button("Download full log (JSONL)", data=log.path.read_bytes() if log.path.exists() else b"", file_name="nrc_queries_audit.jsonl", mime="application/x-ndjson")
    c2.caption(f"{total} record(s) in {log.path}")

    st.markdown("### Record")
    ids = [r["audit_id"] for r in records]
    sel = st.selectbox("Audit id", ids, format_func=lambda i: f"{i} · {next(r for r in records if r['audit_id'] == i).get('question', '')[:80]}")
    rec = next(r for r in records if r["audit_id"] == sel)
    st.markdown(pill(rec.get("status", "")) + pill(f"{rec.get('provider')}/{rec.get('model')}") + pill(rec.get("mode", "")) + pill("support check " + ("on" if rec.get("judge_enabled") else "off")) + pill(f"prompts {rec.get('versions', {}).get('prompts', '?')}") + pill(f"pipeline {rec.get('versions', {}).get('pipeline', '?')}"), unsafe_allow_html=True)
    st.markdown(f"**Question:** {esc(rec.get('question', ''))}")
    if rec.get("error"):
        st.error(rec["error"])

    st.markdown("**Statements**")
    for c in rec.get("claims", []):
        marks = "".join(f"[{n}]" for n in c.get("citations", []))
        st.markdown(f"{badge(c.get('status', ''))} {esc(c.get('text', ''))} {marks}", unsafe_allow_html=True)
        if c.get("judge"):
            st.caption(f"support check: {c['judge'].get('verdict')} — {c['judge'].get('reason', '')}")
        for e in c.get("removed", []):
            st.caption(f"rejected quote ({e.get('chunk_id')}): “{e.get('quote')}” — {e.get('reason')}")

    st.markdown("**Citations**")
    for cit in rec.get("citations", []):
        st.markdown(f"[{cit.get('number')}] {badge(cit.get('status', ''))} {esc(cit.get('tlr_number') or cit.get('doc_id', ''))} · page {cit.get('page')} · “{esc(cit.get('quote', ''))}”", unsafe_allow_html=True)

    with st.expander("Retrieval (what the model saw)"):
        if rec.get("retrieval"):
            st.dataframe(pd.DataFrame(rec["retrieval"]), hide_index=True, use_container_width=True)
    with st.expander("Raw model output"):
        st.code(rec.get("raw_model_output", ""), language=None)
    with st.expander("Full record (JSON)"):
        st.code(json.dumps(rec, indent=2, ensure_ascii=False), language="json")
