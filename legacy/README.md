# Legacy prototype (v1)

These files are redacted copies of the original prototype that this project replaced:

| File | Purpose |
|---|---|
| `01_query.py` | Two-step RAG over a LlamaCloud index with GPT-4o via the Core42 gateway |
| `02_dashboard.py` | First Streamlit dashboard (result cards, PDF open/download) |
| `flow_diagrams_v1.mmd` | Mermaid flow and sequence diagrams of the v1 architecture |
| `requirements_v1.txt` | v1 dependencies |

The API keys that were hard-coded in the v1 scripts have been removed here; they are read from
environment variables (`CORE42_API_KEY`, `LLAMA_CLOUD_API_KEY`, `LLAMA_CLOUD_ORGANIZATION_ID`)
if you ever need to run them again. Because those keys were stored in plain text, rotate them.

The v2 application (`app.py`, `nrc_rag/`) does not depend on anything in this folder.
