"""Methodology page: what is guaranteed, what is not, and how the pipeline works."""

from __future__ import annotations

import streamlit as st

from nrc_rag.config import APP_VERSION, PIPELINE_VERSION
from nrc_rag.llm.prompts import PROMPT_VERSION
from ui import common
from ui.theme import hero, inject_css

METHOD_MD = """
### What "provable" means here

The system never lets the language model speak for itself. Its output is treated as a *proposal* that must survive
deterministic checks before anything is shown:

1. **Closed-world context.** The model only receives passages retrieved from the local index of the NRC reports. It is instructed to
   use nothing else and to abstain when the passages do not answer the question.
2. **Claims must carry verbatim quotes.** Every statement must cite a passage id and copy a verbatim excerpt. With Claude this is
   enforced by the API's citations feature (the API returns the exact cited spans); with other providers the model returns strict JSON.
3. **Deterministic verification.** Each quote is string-matched against the stored text of the cited passage and page
   (after neutral normalisation of whitespace, quotation marks, dashes and hyphenation). Exact matches are *verified*; near-verbatim
   matches above a strict similarity threshold are flagged; anything else is **rejected**, and a statement with no surviving evidence is
   **withheld** — listed separately, never shown as part of the answer.
4. **Independent support check (optional, on by default).** A second, separately prompted model call reads only the statement and its
   verified quotes and judges whether the quotes actually entail the statement. Statements judged *not supported* are withheld;
   *partially supported* ones are flagged.
5. **Page-level proof.** Each citation is located on the source page and the page image is rendered with the passage highlighted, with
   one-click access to the original PDF. Document identity is pinned by SHA-256 hash.
6. **Figures and diagrams.** Figures are extracted with their captions and can be attached as images so the model can read plots and
   schematics. Statements that depend on reading an image are labelled *figure-derived* and shown next to the figure; they are the
   model's visual reading and must be checked against the displayed figure. Optional AI-generated figure descriptions are clearly
   labelled and only used for retrieval.
7. **Audit trail.** Every question writes an append-only record: retrieval scores, the exact passages sent to the model, raw model
   output, every quote check, judge verdicts, final displayed claims, model and prompt versions, and timings.

### What is guaranteed and what is not

| Guaranteed (mechanically enforced) | Not guaranteed |
|---|---|
| Every displayed statement has at least one quote that exists verbatim (or near-verbatim, flagged) in the cited document at the cited page. | That the quote *means* what the statement says. The support check reduces this risk; the highlighted page lets a reader confirm it in seconds. |
| Nothing outside the retrieved passages is presented as sourced fact. | That the *best* passage was retrieved. Retrieval can miss; the answer is then "not found" or partial, never invented. |
| Withheld statements are visible as withheld, with the reason. | Figure-derived readings (values read off a plot) — these are labelled and require visual confirmation. |
| Every answer is reproducible from its audit record (same passages, same output, same checks). | Numerical reasoning: the model is instructed not to compute; verify any arithmetic yourself. |

"99.99 % hallucination-free" is therefore delivered as a *structural* property — unverifiable text cannot reach the answer — rather than as
a statistical claim about the model. The residual risk is concentrated in (a) misreading of a genuine quote and (b) figure-derived
statements, both of which are surfaced for human review rather than hidden.

### Pipeline

`PDF (PyMuPDF) → page text with block geometry, tables (markdown), figures (PNG + caption), outline sections → token-aware chunks
that are verbatim page slices → local embeddings (Jina v2) in ChromaDB + BM25 → reciprocal-rank fusion → cross-encoder re-ranking →
LLM (claims + quotes) → deterministic quote verification → support check → page highlight → audit record`

All indexing and retrieval run locally. Only the retrieved passages (and, if enabled, figure images) are sent to the configured
model provider; the audit record lists exactly which ones.
"""


def render() -> None:
    inject_css()
    st.markdown(hero(common.logo_b64(), "Methodology & Guarantees", "How answers are grounded, verified and audited — and the limits of those guarantees."), unsafe_allow_html=True)
    st.markdown(METHOD_MD)
    s = common.settings()
    st.markdown("### Current configuration")
    prov = s.selected_provider()
    model = {"anthropic": s.anthropic_model, "openai": s.openai_model, "google": s.google_model}.get(prov or "", "")
    rows = {
        "Application version": APP_VERSION,
        "Pipeline version": PIPELINE_VERSION,
        "Prompt version": PROMPT_VERSION,
        "Default provider / model": f"{prov or 'none configured'} / {model}",
        "Providers with keys": ", ".join(s.available_providers()) or "none",
        "Embedding model": s.embedding_model,
        "Re-ranker": s.reranker_model if s.enable_reranker else "disabled",
        "Retrieval": f"dense top-{s.top_k_dense} + BM25 top-{s.top_k_bm25} → RRF → rerank {s.rerank_candidates} → final {s.top_k_final}",
        "Chunking": f"{s.chunk_tokens} tokens, {s.chunk_overlap_tokens} overlap, page-bounded verbatim slices",
        "Fuzzy-match threshold": f"{s.fuzzy_threshold:.2f} (below this a quote is rejected)",
        "Minimum quote length": f"{s.min_quote_words} words / {s.min_quote_chars} characters",
        "Support check default": "on" if s.enable_support_check else "off",
        "Index directory": str(s.index_dir),
        "Audit directory": str(s.audit_dir),
    }
    st.table({"Setting": list(rows.keys()), "Value": list(rows.values())})
