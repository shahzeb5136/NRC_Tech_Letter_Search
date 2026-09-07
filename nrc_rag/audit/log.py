"""Append-only JSONL audit trail.

Every question produces one record containing: the question, retrieval results
with scores, the exact excerpts sent to the model, the raw model output, every
quote check, the final displayed claims, model/provider identifiers, prompt and
pipeline versions, document hashes and timings. Records are never modified.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

from nrc_rag.utils import utc_now_iso

_LOCK = threading.Lock()


class AuditLog:
    def __init__(self, audit_dir: Path) -> None:
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.audit_dir / "queries.jsonl"

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:16]

    def append(self, record: dict) -> str:
        record = dict(record)
        record.setdefault("audit_id", self.new_id())
        record.setdefault("recorded_at", utc_now_iso())
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _LOCK:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return record["audit_id"]

    def read_recent(self, n: int = 50) -> list[dict]:
        if not self.path.exists():
            return []
        with _LOCK:
            with open(self.path, "r", encoding="utf-8") as f:
                tail = deque(f, maxlen=n)
        out = []
        for line in tail:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        out.reverse()
        return out

    def get(self, audit_id: str) -> Optional[dict]:
        if not self.path.exists():
            return None
        with _LOCK:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if audit_id in line:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("audit_id") == audit_id:
                            return rec
        return None

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
