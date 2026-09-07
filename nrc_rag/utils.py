"""Small shared helpers."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


_WS_RE = re.compile(r"\s+")
_QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "­": "",  # soft hyphen
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "�": "",
})


def normalize_text(s: str) -> str:
    """Normalisation used for quote matching.

    Unicode NFKC, typographic quotes/dashes folded to ASCII, ligatures expanded,
    whitespace collapsed, lower-cased. Applied identically to quotes and sources so
    that a match is a claim of verbatim equality up to typography.
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_QUOTE_MAP)
    s = _WS_RE.sub(" ", s)
    return s.strip().lower()


def dehyphenate(s: str) -> str:
    """Join words hyphenated at line ends, e.g. 'cor-' + newline + 'rosion' -> 'corrosion'."""
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)


_BM25_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[\-\x27][a-z0-9]+)*")


def tokenize_for_bm25(text: str) -> list[str]:
    return _BM25_TOKEN_RE.findall(text.lower())


def batched(items: list, n: int) -> Iterable[list]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def short_hash(s: str, n: int = 10) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def smart_title_case(s: str) -> str:
    """Title-case an ALL-CAPS title while preserving acronyms and lower-casing small words."""
    small = {"of", "and", "for", "in", "the", "to", "a", "an", "on", "at", "by", "with", "or", "from", "vs", "into", "as"}
    acronyms = {
        "NRC", "NUREG", "LOCA", "ASME", "BWR", "PWR", "MSR", "RCP", "UQ", "AI", "ML", "IST", "ACM", "RIM", "EMDA", "TRISO",
        "LWR", "HTGR", "SFR", "EPRI", "INL", "PNNL", "ANL", "LANL", "SNL", "ORNL", "DOE", "CFR", "ISI", "NDE", "SSC", "PRA",
        "CMC", "SIC", "MC&A", "LOCAS", "RES", "DE", "REB", "TLR", "US", "U.S.", "III", "VIII", "II", "IV", "V", "VI", "VII",
        "HALEU", "FHR", "SMR", "PSA", "RCS", "ECCS", "GPWR", "MSRE", "XLPR",
    }
    words = s.split()
    out = []
    for i, w in enumerate(words):
        core = re.sub(r"[^A-Za-z0-9&.]", "", w)
        if not core:
            out.append(w)
        elif w.lower() in small and i != 0:
            out.append(w.lower())
        elif core.upper() in acronyms or any(ch.isdigit() for ch in core):
            out.append(w.upper() if core.upper() in acronyms else w)
        else:
            out.append(w[0].upper() + w[1:].lower())
    return " ".join(out)
