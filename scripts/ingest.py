"""Build or update the local index from the PDFs in the data folder.

Examples
--------
    python scripts/ingest.py                       # incremental: only new/changed PDFs
    python scripts/ingest.py --force               # rebuild everything
    python scripts/ingest.py --only ML25002A104    # one document
    python scripts/ingest.py --warm-models         # pre-download the reranker so first query is fast
    python scripts/ingest.py --describe-figures --limit 20   # AI descriptions of figures (uses the configured LLM provider)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrc_rag.config import get_settings  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-index documents even if unchanged")
    parser.add_argument("--only", nargs="*", help="doc ids (file stems) to process")
    parser.add_argument("--no-figures", action="store_true", help="skip figure extraction")
    parser.add_argument("--describe-figures", action="store_true", help="generate AI descriptions for figures lacking one")
    parser.add_argument("--limit", type=int, default=None, help="max figures to describe in this run")
    parser.add_argument("--warm-models", action="store_true", help="download/load embedding + reranker models and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()

    def say(msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    if args.warm_models:
        from nrc_rag.index.embeddings import Embedder, try_load_reranker

        say(f"loading embedding model {settings.embedding_model}")
        Embedder(settings.embedding_model, settings.embedding_max_seq)
        if settings.enable_reranker:
            say(f"loading reranker {settings.reranker_model}")
            ok = try_load_reranker(settings.reranker_model)
            say("reranker ready" if ok else "reranker unavailable (retrieval will use fused ranking only)")
        say("models ready")
        return 0

    if args.describe_figures:
        from nrc_rag.ingest.pipeline import describe_figures
        from nrc_rag.llm import get_provider

        provider = get_provider(settings)
        if provider is None:
            say("No LLM provider configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY or GOOGLE_API_KEY in .env")
            return 2
        n = describe_figures(settings, provider, limit=args.limit, doc_ids=args.only, progress=say)
        say(f"described {n} figure(s)")
        return 0

    from nrc_rag.ingest.pipeline import ingest

    report = ingest(settings, force=args.force, only=args.only, extract_figures=not args.no_figures, progress=say)
    say(
        f"finished in {report.elapsed_seconds:.0f}s: processed={len(report.processed)} skipped={len(report.skipped)} "
        f"failed={len(report.failed)} chunks_added={report.chunks_added} figures_added={report.figures_added}"
    )
    for doc_id, err in report.failed.items():
        say(f"  FAILED {doc_id}: {err}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
