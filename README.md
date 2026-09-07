# NRC Technical Letter Query System

Grounded, citation-verified question answering over U.S. NRC Technical Letter Reports (TLR-RES/DE/REB series),
including their figures and diagrams. Built for a setting where an unsupported statement is worse than no answer:
**every statement shown to the user carries a verbatim quote that was deterministically verified against the source
page, located on that page, and recorded in an audit trail. Anything that cannot be verified is withheld.**

<p align="center"><img src="logo.png" height="64"></p>

## Contents

1. [What you get](#what-you-get)
2. [The grounding guarantee](#the-grounding-guarantee)
3. [Quick start](#quick-start)
4. [Configuration](#configuration)
5. [Architecture](#architecture)
6. [Figures and diagrams](#figures-and-diagrams)
7. [Audit trail and exports](#audit-trail-and-exports)
8. [Evaluation](#evaluation)
9. [Tests](#tests)
10. [Publishing to GitHub](#publishing-to-github)
11. [Deployment](#deployment)
12. [Limitations](#limitations)
13. [Project layout](#project-layout)

## What you get

| Page | Purpose |
|---|---|
| **Ask** | Natural-language questions → verified answer. Each statement has numbered citations; each citation shows the verbatim quote, its verification status, the rendered source page with the passage highlighted, and a one-click open of the PDF. Withheld statements are listed separately with the reason. Search-only mode works without any API key. |
| **Document Library** | Every indexed report with title, TLR number, date, laboratory, page/passage/figure counts and SHA-256 hash; outline, figure gallery, page viewer with extracted text. |
| **Audit Trail** | Append-only log of every question: retrieval scores, the exact passages sent to the model, raw model output, every quote check, support-check verdicts, displayed claims, model and prompt versions, timings. Downloadable. |
| **Methodology & Guarantees** | What is mechanically guaranteed, what is not, and the live configuration. |

## The grounding guarantee

The language model is never the last word. Its output is a *proposal* that must survive deterministic checks:

1. **Closed world.** The model only sees passages retrieved from the local index and is instructed to abstain otherwise.
2. **Claims carry verbatim quotes.** Every statement must cite a passage id and copy a verbatim excerpt.
   With Claude this is enforced by the Messages API *citations* feature (the API returns the exact cited spans);
   with OpenAI-compatible endpoints and Gemini the model returns strict JSON.
3. **Deterministic verification** (`nrc_rag/verify/quote_verifier.py`). Each quote is matched against the stored text of
   the cited passage and page after neutral normalisation (whitespace, quotation marks, dashes, ligatures, line-end
   hyphenation, table markup). Exact match → *verified*. Similarity ≥ 0.92 → *near-verbatim* (flagged).
   Otherwise the quote is **rejected**; a statement with no surviving quote is **withheld** and shown only in the
   withheld list.
4. **Independent support check** (optional, on by default). A second, separately prompted model call sees only the
   statement and its verified quotes and judges entailment. *Not supported* → withheld. *Partially supported* → flagged.
5. **Page-level proof.** Each citation is located on the page with PyMuPDF text search and the page is rendered with the
   passage highlighted. Documents are pinned by SHA-256.
6. **Figures** are extracted with captions and can be sent as images; statements that depend on reading an image are
   labelled *figure-derived* and displayed next to the figure for visual confirmation.
7. **Audit record** for every question.

"99.99 % hallucination-free" is delivered as a **structural property** — unverifiable text cannot reach the answer — not
as a statistical claim about a model. Residual risk is concentrated in (a) a genuine quote being misread by the model and
(b) figure-derived readings; both are surfaced for human review (support check, highlighted page, displayed figure).

## Quick start

Requirements: Python 3.11+ (developed on 3.12), ~4 GB RAM for the local embedding model (CPU is fine), the PDFs under `Data/`.

```bash
pip install -r requirements.txt
```

Configure secrets (never committed):

```bash
copy .env.example .env      # Windows   (cp .env.example .env on macOS/Linux)
```

Put at least one key in `.env`: `ANTHROPIC_API_KEY` (recommended), or `OPENAI_API_KEY` (+ `OPENAI_BASE_URL` for a
gateway such as Core42 or a local vLLM/Ollama server), or `GOOGLE_API_KEY`. Without a key the app runs in search-only mode.

Build the local index (incremental; only new or changed PDFs are processed):

```bash
python scripts/ingest.py
```

Run the app:

```bash
streamlit run app.py
```

First launch downloads the local embedding model (`jinaai/jina-embeddings-v2-base-en`) and the cross-encoder re-ranker
if they are not cached; `python scripts/ingest.py --warm-models` does this ahead of time.

Optional — AI descriptions of figures for better diagram retrieval (uses the configured provider; cached by image hash so
they are never regenerated):

```bash
python scripts/ingest.py --describe-figures --limit 50
```

## Configuration

All settings come from `.env` / environment variables (see `.env.example`). Key ones:

| Variable | Default | Meaning |
|---|---|---|
| `NRC_LLM_PROVIDER` | `auto` | `anthropic`, `openai`, `google`, or `auto` (first configured key in that order) |
| `ANTHROPIC_MODEL` / `ANTHROPIC_EFFORT` | `claude-opus-5` / `high` | Claude model and reasoning effort |
| `OPENAI_BASE_URL` / `OPENAI_MODEL` | — / `gpt-4o` | Any OpenAI-compatible endpoint |
| `GOOGLE_MODEL` | `gemini-2.5-pro` | Gemini model |
| `NRC_TOP_K_FINAL` | `10` | Passages sent to the model after fusion and re-ranking |
| `NRC_MAX_FIGURES` | `4` | Figure images attached per question |
| `NRC_FUZZY_THRESHOLD` | `0.92` | Below this similarity a quote is rejected |
| `NRC_ENABLE_SUPPORT_CHECK` | `true` | Second-pass entailment check |
| `NRC_INDEX_DIR` / `NRC_AUDIT_DIR` | `index/` / `audit/` | Where the index and audit log live |
| `NRC_EMBEDDING_MAX_POSITIONS` | `1024` | Caps the embedder's declared position count. See *Deployment* — this is the difference between 1.6 GB and 4.7 GB of memory |
| `NRC_TORCH_THREADS` | `2` | Torch thread cap, keeps memory predictable on small hosts |

Keys can also be provided through `.streamlit/secrets.toml` (git-ignored) for Streamlit Community Cloud.

## Architecture

```
Data/*.pdf
   │  PyMuPDF: page text with block geometry, tables → markdown, figures → PNG + caption,
   │  outline → section per chunk, running headers/footers removed, TOC pages skipped
   ▼
chunks  (page-bounded verbatim slices, ~380 tokens, 50 overlap; kinds: text | table | figure)
   │
   ├─► vectors.npz (local, exact cosine) ← Jina v2 embeddings (local, CPU)
   └─► BM25 (rank-bm25)             ┐
                                    ├─► reciprocal-rank fusion → cross-encoder re-rank → top-k
   question ─► query embedding ─────┘
                                    │
                                    ▼
                 LLM provider (Claude citations | OpenAI-compatible JSON | Gemini JSON)
                                    │ claims + verbatim quotes + chunk ids
                                    ▼
                 quote verifier (deterministic) → support check (model) → page highlight
                                    │
                                    ▼
                 Streamlit UI  +  audit/queries.jsonl  +  Markdown / JSON export
```

Everything up to the model call runs locally; only the retrieved passages (and, if enabled, figure images) leave the
machine, and the audit record lists exactly which ones.

Provider notes:

* **Anthropic Claude** (`nrc_rag/llm/anthropic_provider.py`) — each passage is a `document` block with
  `citations: {enabled: true}`; the response's cited spans become the claims' evidence and are re-verified locally.
  Adaptive thinking and `output_config.effort` are used; server-side refusal fallbacks are available but off by default
  so that the audit trail always names the model that answered (`ANTHROPIC_ENABLE_FALLBACKS=true` to enable).
* **OpenAI-compatible** (`openai_provider.py`) — temperature 0, JSON mode when the gateway supports it (falls back
  automatically), vision through `image_url` data URLs. Written for gateways such as Core42; the Core42 key available during
  development had no remaining quota, so this path has only been exercised against the request/response contract, not a live model.
* **Google Gemini** (`google_provider.py`) — structured JSON output through `response_schema`, images as inline parts.
  Verified end to end with `gemini-2.5-pro` (see `evaluation/`). Note that Gemini 2.5 spends thinking tokens from
  `max_output_tokens`, so budgets are set generously.

## Figures and diagrams

* Figures are detected per page from embedded raster images and clustered vector drawings; sub-panels sharing a caption
  are merged; repeated logos and uncaptioned decorations are dropped. Each figure is cached to `index/figures/…` at
  150 dpi with its caption and indexed as a passage. Those PNGs are a cache, not part of the index: when the file is
  absent the crop is re-rendered from the PDF using the stored bounding box, which is why a deployment can ship the
  index without ~70 MB of images. Each figure is indexed as a passage (`<doc>:p<page>:f<n>`) using caption + surrounding text.
* When a figure passage is retrieved, its image can be attached to the model request so plots and schematics can be read.
  Resulting statements are labelled **figure-derived** and shown next to the figure.
* `--describe-figures` adds a clearly labelled AI description to the figure passage (retrieval only, cached by image hash).

## Audit trail and exports

`audit/queries.jsonl` receives one JSON record per question (never modified). Each record is self-contained:
question, filters, provider/model, versions (app, pipeline, prompts), retrieval results with dense/BM25/re-rank scores,
passage ids sent to the model, raw model output, every quote check with status/score/reason/page rectangles, support-check
verdicts, displayed and withheld claims, token usage and timings. The Ask page exports the same record as JSON and the
answer as Markdown; the Audit page lets you inspect or download everything.

## Evaluation

`scripts/evaluate.py` runs `evaluation/questions.yaml` (answerable questions with expected documents plus deliberately
unanswerable ones) through the full pipeline and reports: answered rate, expected-document citation rate, **abstention rate
on unanswerable questions**, share of generated claims that survived verification, and exact/near-verbatim/rejected quote
rates. Reports are written next to the question file.

```bash
python scripts/evaluate.py --provider openai
```

## Tests

```bash
python -m pytest tests -q
```

Covers the quote verifier (typography, hyphenation, tables, fabricated quotes, page-level matches), model-output parsing,
the chunker's verbatim-slice/provenance guarantee, and the conversion of Claude citation blocks into claims (API stubbed).

## Publishing to GitHub

`.gitignore` already excludes `.env`, `.streamlit/secrets.toml`, the `audit/` log, caches, Google-Drive artefacts and
the un-redacted v1 scripts. The index **is** committed (see *Deployment*) apart from its derivable parts. Before the first push:

1. Confirm no key is present: `git grep -n -E "sk-ant-|llx-|api_key ?= ?\"" -- ':!legacy/README.md'` should print nothing.
2. Decide whether `Data/` (≈150 MB of public NRC PDFs) belongs in the repository or should be downloaded separately
   (accession numbers are in the Library page; ADAMS URL pattern `https://www.nrc.gov/docs/MLyyxx/MLyyxxxxxxxx.pdf`).
3. **Rotate** the Core42 and LlamaCloud keys that were hard-coded in the v1 scripts; they lived in plain text.

## Deployment

The app runs on a small host (it is deployed on Streamlit Community Cloud, ~2.7 GB of memory) because of two
decisions that are easy to get wrong:

**The index is committed.** A cloud host has an ephemeral filesystem, so an index built at deploy time is lost on
every restart, and rebuilding it takes about 15 minutes. Three files are tracked, 20 MB in total:

| File | Size | Why |
|---|---|---|
| `index/catalog.db` | 10 MB | Page text and chunks — the source of truth the verifier matches quotes against |
| `index/vectors.npz` | 10 MB | Embeddings, as a plain normalised float32 array |
| `index/manifest.json` | 12 KB | What was indexed, with file hashes |

Not committed, because both are derivable: `index/figures/` (~70 MB of PNGs, re-cropped from the PDFs on demand)
and the figure-description cache.

**The embedder's position count is capped.** `jina-embeddings-v2` declares 8192 positions and allocates an
attention bias of that size when it loads — a single ~3.4 GB tensor, against 549 MB of actual weights. Since chunks
are built to 512 tokens, everything above the cap is dead weight. `NRC_EMBEDDING_MAX_POSITIONS=1024` drops peak
memory for a full search from ~4.7 GB to ~1.6 GB and leaves embeddings bit-for-bit identical.

Dense retrieval is a dot product against that array rather than a vector database: at ~3k passages an exact search
is microseconds, and a plain array has no version-portability problem when a prebuilt index is committed and
restored on another machine.

`requirements.txt` pins CPU-only torch through PyTorch's CPU index, otherwise pip installs the CUDA build and the
cold start pays for gigabytes of GPU libraries that are never used.

**Pin the host's Python version to 3.12.** This is the one setting that is not in the repository. PyTorch's CPU wheel
index publishes builds for 3.10-3.13 and none for 3.14, which is the current Streamlit Community Cloud default, so on
3.14 the embedding stack cannot install and search fails while the rest of the app keeps working. On Streamlit
Community Cloud: *Manage app -> Settings -> Advanced -> Python version -> 3.12*, then reboot.

**Provider keys are set in the host, not in the repository.** `.env` is deliberately untracked. On Streamlit Community
Cloud add the key under *Manage app -> Settings -> Secrets*, in TOML form:

```toml
GOOGLE_API_KEY = "..."
```

Without a key the app runs in search-only mode: retrieval, the library, figures and the audit trail all work, but no
answers are generated.

## Limitations

* Retrieval can miss the best passage; the result is then "not found" or a partial answer, never an invented one.
* The verifier proves *presence* of a quote, not *meaning*; the support check and highlighted page address that, and
  figure-derived statements always need visual confirmation.
* The model is told not to compute; check any arithmetic yourself.
* Scanned PDFs without a text layer are not OCR'd (all current reports are born-digital).
* Table extraction is heuristic; complex tables are indexed as markdown and cited by bounding box.

## Project layout

```
app.py                     Streamlit entry point (multi-page navigation)
ui/                        pages: ask, library, audit, methodology; theme; cached resources
nrc_rag/
  config.py                settings (.env), versions
  ingest/pdf_extract.py    PyMuPDF extraction: text+geometry, tables, figures, sections, metadata
  ingest/chunker.py        page-bounded verbatim chunks with bbox + section
  ingest/pipeline.py       incremental indexing, figure descriptions
  index/embeddings.py      local embedder + cross-encoder re-ranker
  index/store.py           SQLite catalog + vectors + manifest
  index/vectors.py         portable .npz vector store with exact cosine search
  index/retriever.py       hybrid retrieval (dense + BM25 + RRF + re-rank)
  llm/                     provider adapters, prompts, answer schema/parsing
  verify/quote_verifier.py deterministic quote verification
  verify/engine.py         end-to-end grounded answer + audit record
  render/page_render.py    locate quotes on pages, render highlights
  render/figures.py        figure images, from cache or re-cropped from the PDF
  audit/log.py             append-only JSONL audit trail
scripts/ingest.py          build/update the index, warm models, describe figures
scripts/evaluate.py        groundedness evaluation harness
evaluation/questions.yaml  evaluation set
tests/                     pytest suite
legacy/                    redacted v1 prototype (for reference only)
Data/                      source PDFs (ML accession numbers)
```
