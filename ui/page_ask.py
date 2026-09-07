"""Ask page: question -> verified, cited answer with page-level proof."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from nrc_rag.verify.engine import STATUS_LABELS, VerifiedAnswer
from ui import common
from ui.theme import BLUE, GOLD, GREEN, MUTED, RED, badge, banner, esc, hero, inject_css, pill

EXAMPLES = [
    "What off-gas treatment technologies are considered for molten salt reactors and what are their main limitations?",
    "How does the reexamination of NUREG-1829 change the LOCA frequency estimates for BWR piping?",
    "What reference electrodes are discussed for electrochemical monitoring in molten salts?",
    "What are the main knowledge gaps for electrical cable degradation in long-term operation?",
    "Which chemical process safety hazards are identified for TRISO fuel fabrication facilities?",
    "What machine learning approaches were investigated for BWR recirculation pump condition monitoring?",
]


def _show_image(data: bytes, caption: str | None = None) -> None:
    try:
        st.image(data, caption=caption, use_container_width=True)
    except TypeError:  # older Streamlit
        st.image(data, caption=caption, use_column_width=True)


def _set_question(text: str) -> None:
    st.session_state["question_input"] = text


def _sidebar(s, docs):
    with st.sidebar:
        st.markdown("### Answer configuration")
        providers = s.available_providers()
        model_names = {"anthropic": s.anthropic_model, "openai": s.openai_model, "google": s.google_model}
        prov_name = None
        if providers:
            default = s.selected_provider() if s.selected_provider() in providers else providers[0]
            prov_name = st.selectbox(
                "Language model",
                providers,
                index=providers.index(default),
                format_func=lambda n: f"{n} · {model_names.get(n, '')}",
                help="Claude uses the native citations API (API-level verbatim spans). Other providers return strict JSON that is verified locally.",
            )
        else:
            st.warning("No API key configured. Add ANTHROPIC_API_KEY, OPENAI_API_KEY or GOOGLE_API_KEY to .env to enable answers. Search-only mode is available.")
        search_only = st.toggle("Search only (no AI synthesis)", value=prov_name is None, disabled=prov_name is None)
        use_judge = st.toggle("Independent support check", value=s.enable_support_check, help="A second, separately prompted model call checks that each verified quote actually supports the statement. Statements judged unsupported are withheld.")
        attach_images = st.toggle("Send figure images to the model", value=True, help="Figures retrieved for the question are attached as images so the model can read diagrams. Figure-derived statements are labelled and must be checked visually.")
        top_k = st.slider("Passages retrieved", min_value=4, max_value=16, value=s.top_k_final, help="Top passages after hybrid retrieval and re-ranking that are sent to the model.")
        chosen = st.multiselect("Restrict to documents", options=[d.doc_id for d in docs], format_func=lambda i: common.doc_label(next(d for d in docs if d.doc_id == i)), placeholder="All indexed documents")
        st.markdown("---")
        st.markdown("### Index")
        stats = common.store().stats()
        c1, c2 = st.columns(2)
        c1.metric("Documents", stats["documents"])
        c2.metric("Pages", stats["pages"])
        c1.metric("Passages", stats["chunks"])
        c2.metric("Figures", stats["figures"])
        rr = common.reranker()
        st.caption(f"Embeddings: {s.embedding_model}\n\nRe-ranker: {s.reranker_model if rr else 'off'}\n\nFigures with AI descriptions: {stats['figures_described']}")
    return prov_name, search_only, use_judge, attach_images, top_k, chosen


def _status_banner(va: VerifiedAnswer) -> None:
    s = va.stats
    if va.status == "error":
        st.markdown(banner(f"⛔ The request failed: {esc(va.error)}", RED), unsafe_allow_html=True)
        return
    if va.status == "not_found":
        st.markdown(banner("⚠️ The indexed documents do not contain a verifiable answer to this question. Nothing was generated beyond the evidence.", GOLD), unsafe_allow_html=True)
        return
    parts = [f"✅ {s['claims_displayed']} of {s['claims_total']} statements are backed by verified quotes"]
    if s["claims_approximate"]:
        parts.append(f"{s['claims_approximate']} near-verbatim")
    if s["claims_figure"]:
        parts.append(f"{s['claims_figure']} figure-derived (check visually)")
    if s["claims_withheld"]:
        parts.append(f"{s['claims_withheld']} withheld")
    color = GREEN if not s["claims_withheld"] and not s["claims_figure"] else GOLD
    label = "Partial answer" if va.status == "partial" else "Answer"
    st.markdown(banner(f"{label} · " + " · ".join(parts), color), unsafe_allow_html=True)


def _claim_html(c) -> str:
    color = {"verified": GREEN, "approximate": GOLD, "figure": BLUE, "connective": MUTED}.get(c.status, MUTED)
    sups = "".join(f"<sup>[{n}]</sup>" for n in c.citation_numbers)
    cls = "nrc-claim connective" if c.status == "connective" else "nrc-claim"
    b = "" if c.status in ("verified", "connective") else " " + badge(c.status, STATUS_LABELS.get(c.status, c.status))
    return f'<div class="{cls}" style="border-left-color:{color}">{esc(c.text)}{sups}{b}</div>'


def _render_citation(cit, s) -> None:
    ch = cit.check
    header = f"[{cit.number}] {cit.tlr_number or ch.doc_id} · page {ch.page_number}"
    if ch.section:
        header += f" · {ch.section[:60]}"
    if ch.kind != "text":
        header += f" · {ch.kind}"
    with st.expander(header, expanded=False):
        st.markdown(f"**{esc(cit.doc_title)}**  \n<span class='nrc-meta'>{esc(ch.doc_id)} · {esc(cit.tlr_number)} · page {ch.page_number}{(' · § ' + esc(ch.section)) if ch.section else ''} · chunk <span class='nrc-kbd'>{esc(ch.chunk_id)}</span></span>", unsafe_allow_html=True)
        cls = "nrc-quote fuzzy" if ch.status == "fuzzy" else "nrc-quote"
        st.markdown(f'<div class="{cls}">“{esc(ch.quote)}”</div>', unsafe_allow_html=True)
        st.markdown(badge(ch.status, {"exact": "Verbatim match", "fuzzy": "Near-verbatim match"}.get(ch.status, ch.status)) + f" <span class='nrc-meta'>{esc(ch.reason)}</span>", unsafe_allow_html=True)
        if ch.status == "fuzzy" and ch.matched_text:
            st.caption(f"Closest source text: “{ch.matched_text}”")
        fig_png = common.figure_image(cit.figure_id) if cit.figure_id else None
        if fig_png:
            _show_image(fig_png, caption=cit.caption or "Figure")
            st.caption("Figure-derived statements are the model's reading of this image. Verify against the figure.")
        cols = st.columns([1, 1, 3])
        show_page = cols[0].checkbox("Show page", value=True, key=f"showpage_{cit.number}_{ch.chunk_id}")
        if cols[1].button("Open PDF", key=f"open_{cit.number}_{ch.chunk_id}", help="Open the source PDF in the system viewer"):
            ok, msg = common.open_in_system_viewer(cit.pdf_path)
            (st.success if ok else st.error)(msg)
        if show_page and cit.pdf_path and Path(cit.pdf_path).exists():
            rects = tuple(tuple(r) for r in ch.rects)
            try:
                png = common.page_png(cit.pdf_path, ch.page_number, rects, s.render_dpi, ch.approximate_location)
                cap = f"{ch.doc_id} · page {ch.page_number}" + (" · highlighted region is approximate" if ch.approximate_location else " · quoted passage highlighted")
                _show_image(png, caption=cap)
            except Exception as exc:
                st.warning(f"Page could not be rendered: {exc}")


def _render_answer(va: VerifiedAnswer, s) -> None:
    _status_banner(va)
    if va.status == "error":
        return
    head = st.columns([3, 2])
    with head[0]:
        st.markdown(pill(f"{va.provider} · {va.model}") + pill("API citations" if va.mode == "citations" else "Strict JSON + local verification") + pill("Support check on" if va.judge_enabled else "Support check off") + pill(f"audit {va.audit_id}"), unsafe_allow_html=True)
    with head[1]:
        t = va.timings
        st.caption(f"retrieval {t.get('retrieval_s', 0):.1f}s · generation {t.get('generation_s', 0):.1f}s · verification {t.get('verification_s', 0):.1f}s · support check {t.get('support_check_s', 0):.1f}s · total {t.get('total_s', 0):.1f}s")

    if va.displayed_claims:
        st.markdown("#### Answer")
        st.markdown("".join(_claim_html(c) for c in va.displayed_claims), unsafe_allow_html=True)
    if va.notes:
        st.markdown("<span class='nrc-meta'>Model notes (scope caveats, not verified):</span>", unsafe_allow_html=True)
        for n in va.notes:
            st.markdown(f"<div class='nrc-claim connective' style='border-left-color:{MUTED}'>{esc(n)}</div>", unsafe_allow_html=True)
    if va.withheld_claims:
        with st.expander(f"Withheld statements ({len(va.withheld_claims)}) — failed verification, NOT part of the answer", expanded=False):
            for c in va.withheld_claims:
                reason = (c.judge or {}).get("reason") if c.status == "unsupported" else (c.removed[0].reason if c.removed else (c.notes[0] if c.notes else "no verifiable evidence"))
                st.markdown(f"<div class='nrc-withheld'><s>{esc(c.text)}</s><br/><span class='nrc-meta'>{badge(c.status, STATUS_LABELS.get(c.status, c.status))} {esc(reason or '')}</span></div>", unsafe_allow_html=True)
                for r in c.removed:
                    st.markdown(f'<div class="nrc-quote failed">“{esc(r.quote)}” <span class="nrc-meta">({esc(r.chunk_id)}: {esc(r.reason)})</span></div>', unsafe_allow_html=True)

    if va.citations:
        st.markdown("#### Evidence")
        st.caption("Every citation below was matched deterministically against the stored text of the cited page. Open a citation to see the quote highlighted on the actual page.")
        for cit in va.citations:
            _render_citation(cit, s)

    with st.expander(f"Retrieved context ({len(va.retrieval)} passages sent to the model)", expanded=False):
        if va.retrieval:
            df = pd.DataFrame(va.retrieval)[["rank", "doc_id", "page", "kind", "section", "rerank_score", "dense_rank", "bm25_rank", "chunk_id"]]
            st.dataframe(df, hide_index=True, use_container_width=True)
            chunk_rows = common.store().get_chunks(va.context_ids)
            pick = st.selectbox("Show passage", va.context_ids, key=f"ctx_{va.audit_id}")
            if pick in chunk_rows:
                st.code(chunk_rows[pick].text, language=None)

    cols = st.columns([1, 1, 4])
    cols[0].download_button("Export answer (Markdown)", data=va.to_markdown().encode("utf-8"), file_name=f"nrc_answer_{va.audit_id}.md", mime="text/markdown", key=f"md_{va.audit_id}")
    cols[1].download_button("Export audit record (JSON)", data=json.dumps(va.to_record(), indent=2, ensure_ascii=False).encode("utf-8"), file_name=f"nrc_audit_{va.audit_id}.json", mime="application/json", key=f"json_{va.audit_id}")
    if va.usage:
        st.caption("Token usage: " + ", ".join(f"{k} {v}" for k, v in va.usage.items() if v is not None))


def _render_search(results, s) -> None:
    if not results:
        st.info("No passages matched.")
        return
    st.markdown(f"#### Top {len(results)} passages")
    for r in results:
        c = r.chunk
        header = f"{r.rank}. {r.tlr_number or c.doc_id} · page {c.page_number}" + (f" · {c.section[:60]}" if c.section else "") + (f" · {c.kind}" if c.kind != "text" else "")
        with st.expander(header, expanded=r.rank <= 3):
            st.markdown(f"**{esc(r.doc_title)}** <span class='nrc-meta'>· {esc(c.chunk_id)} · rerank {r.rerank_score if r.rerank_score is None else round(r.rerank_score, 2)} · dense #{r.dense_rank or '-'} · bm25 #{r.bm25_rank or '-'}</span>", unsafe_allow_html=True)
            st.write(c.text)
            if c.kind == "figure" and c.figure_id:
                fig = common.store().get_figure(c.figure_id)
                fig_png = common.figure_image(c.figure_id)
                if fig_png:
                    _show_image(fig_png, caption=(fig.caption if fig else "") or "Figure")
            doc = common.store().get_document(c.doc_id)
            if doc and st.button("Open PDF", key=f"sopen_{c.chunk_id}"):
                ok, msg = common.open_in_system_viewer(doc.path)
                (st.success if ok else st.error)(msg)


def render() -> None:
    inject_css()
    s = common.settings()
    st.markdown(hero(common.logo_b64(), "NRC Technical Letter Query System", "Grounded question answering over NRC Technical Letter Reports — every statement carries a verbatim, page-located citation, verified before it is shown."), unsafe_allow_html=True)

    if not common.index_ready():
        st.warning("The local index is empty. Build it first:")
        st.code("python scripts/ingest.py", language="bash")
        st.stop()

    search_error = common.retrieval_status()
    if search_error:
        st.error(
            "**Search is unavailable on this host: the embedding model could not be loaded.**\n\n"
            "The rest of the app still works — the Document Library, Audit Trail and Methodology pages "
            "read the index directly and need no model."
        )
        with st.expander("What went wrong, and how to fix it"):
            st.code(search_error, language=None)
            st.markdown(
                "This is the installed environment rather than the code, and it is almost always a dependency "
                "version that drifted ahead of what the embedding model expects.\n\n"
                "`requirements.txt` pins the combination that is known to work. Reinstall against it:\n\n"
                "```\npip install -r requirements.txt\n```\n\n"
                "On a hosted deployment, reboot the app so it reinstalls, and check the build log for the "
                "package that failed."
            )
        st.stop()

    docs = common.store().list_documents()
    prov_name, search_only, use_judge, attach_images, top_k, chosen = _sidebar(s, docs)

    st.text_area("Question", key="question_input", height=96, placeholder="e.g. What sorbent materials are evaluated for iodine capture in molten salt reactor off-gas systems?")
    ex_cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        ex_cols[i % 3].button(ex if len(ex) <= 92 else ex[:89] + "…", key=f"ex_{i}", on_click=_set_question, args=(ex,), use_container_width=True)

    run_cols = st.columns([1, 5])
    ask = run_cols[0].button("Ask", type="primary", use_container_width=True)
    run_cols[1].caption("Answers are generated only from the retrieved passages, then every quote is checked against the stored page text. Unverifiable statements are withheld, never shown as fact.")

    question = (st.session_state.get("question_input") or "").strip()
    if ask:
        if not question:
            st.warning("Enter a question first.")
        elif search_only or prov_name is None:
            with st.spinner("Searching the index…"):
                results = common.engine(None).search_only(question, top_k=top_k, doc_ids=chosen or None)
            st.session_state["last_search"] = results
            st.session_state.pop("last_answer", None)
        else:
            with st.status("Working…", expanded=True) as status:
                eng = common.engine(prov_name)
                va = eng.answer(question, doc_ids=chosen or None, top_k=top_k, use_judge=use_judge, attach_images=attach_images, progress=lambda m: status.write(m))
                status.update(label="Done" if va.status != "error" else "Failed", state="complete" if va.status != "error" else "error", expanded=False)
            st.session_state["last_answer"] = va
            st.session_state.pop("last_search", None)

    if "last_answer" in st.session_state:
        st.markdown("---")
        _render_answer(st.session_state["last_answer"], s)
    elif "last_search" in st.session_state:
        st.markdown("---")
        _render_search(st.session_state["last_search"], s)

    st.markdown("<div class='nrc-footer'>Local index · answers audited to <code>audit/queries.jsonl</code> · verify figure-derived statements against the displayed figure</div>", unsafe_allow_html=True)
