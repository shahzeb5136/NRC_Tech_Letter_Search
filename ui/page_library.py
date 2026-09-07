"""Document library: what is indexed, with outline, figures and a page viewer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui import common
from ui.theme import esc, hero, inject_css, pill


def _show_image(data: bytes, caption: str | None = None) -> None:
    try:
        st.image(data, caption=caption, use_container_width=True)
    except TypeError:
        st.image(data, caption=caption, use_column_width=True)


def render() -> None:
    inject_css()
    s = common.settings()
    st.markdown(hero(common.logo_b64(), "Document Library", "Indexed NRC Technical Letter Reports with content hashes, outlines and extracted figures."), unsafe_allow_html=True)
    if not common.index_ready():
        st.warning("The local index is empty. Run `python scripts/ingest.py` first.")
        st.stop()

    store = common.store()
    docs = store.list_documents()
    manifest = store.read_manifest()
    st.caption(f"Pipeline {manifest.get('pipeline_version', '?')} · embeddings {manifest.get('embedding_model', '?')} · last updated {manifest.get('updated_at', '?')}")

    df = pd.DataFrame(
        [
            {
                "Accession": d.doc_id,
                "Report": d.tlr_number,
                "Title": d.title,
                "Date": d.report_date,
                "Organization": d.organization,
                "Pages": d.page_count,
                "Passages": d.chunk_count,
                "Figures": d.figure_count,
                "SHA-256": d.sha256[:16] + "…",
            }
            for d in docs
        ]
    )
    st.dataframe(df, hide_index=True, use_container_width=True, height=min(600, 40 + 36 * len(df)))

    st.markdown("### Document details")
    sel = st.selectbox("Select a document", [d.doc_id for d in docs], format_func=lambda i: common.doc_label(next(d for d in docs if d.doc_id == i)))
    d = next(x for x in docs if x.doc_id == sel)
    st.markdown(f"**{esc(d.title)}**", unsafe_allow_html=True)
    st.markdown(pill(d.doc_id) + pill(d.tlr_number or "no TLR number") + pill(d.report_date or "date n/a") + pill(d.organization or "organization n/a") + pill(f"{d.page_count} pages"), unsafe_allow_html=True)
    st.markdown(f"<span class='nrc-meta'>SHA-256 <span class='nrc-kbd'>{esc(d.sha256)}</span> · indexed {esc(d.ingested_at)} · {esc(d.path)}</span>", unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("Open PDF", key="lib_open"):
        ok, msg = common.open_in_system_viewer(d.path)
        (st.success if ok else st.error)(msg)
    if c2.checkbox("Prepare download", key="lib_dl_prep"):
        data = common.file_bytes(d.path)
        if data:
            st.download_button("Download PDF", data=data, file_name=Path(d.path).name, mime="application/pdf", key="lib_dl")

    tab_outline, tab_figs, tab_pages = st.tabs(["Outline", "Figures", "Page viewer"])
    with tab_outline:
        if d.toc:
            lines = []
            for lvl, title, page in d.toc:
                lines.append(f"{'&nbsp;' * 4 * (int(lvl) - 1)}{esc(str(title))} <span class='nrc-meta'>· p.{page}</span>")
            st.markdown("<br/>".join(lines), unsafe_allow_html=True)
        else:
            st.info("This PDF has no embedded outline; sections were detected from numbered headings.")
    with tab_figs:
        figs = store.list_figures(d.doc_id)
        if not figs:
            st.info("No figures were extracted from this document.")
        else:
            per_page = 9
            n_pages = (len(figs) + per_page - 1) // per_page
            pg = st.number_input("Gallery page", min_value=1, max_value=max(1, n_pages), value=1, step=1, key="fig_gallery_page")
            subset = figs[(pg - 1) * per_page : pg * per_page]
            cols = st.columns(3)
            for i, f in enumerate(subset):
                with cols[i % 3]:
                    if Path(f.image_path).exists():
                        _show_image(Path(f.image_path).read_bytes(), caption=f"p.{f.page_number} · {f.caption[:120] if f.caption else 'no caption detected'}")
                    if f.description:
                        with st.expander("AI description"):
                            st.write(f.description)
                            st.caption(f"model {f.description_model} · {f.described_at}")
    with tab_pages:
        pn = st.number_input("Page", min_value=1, max_value=max(1, d.page_count), value=1, step=1, key="lib_page")
        page = store.get_page(d.doc_id, int(pn))
        if page and page.section:
            st.caption(f"Section: {page.section_path or page.section}")
        try:
            png = common.page_png(d.path, int(pn), tuple(), s.render_dpi, False)
            _show_image(png, caption=f"{d.doc_id} · page {int(pn)}")
        except Exception as exc:
            st.warning(f"Page could not be rendered: {exc}")
        if page:
            with st.expander("Extracted text for this page"):
                st.code(page.text or "(no text)", language=None)
