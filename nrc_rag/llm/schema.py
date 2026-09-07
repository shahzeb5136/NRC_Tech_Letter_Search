"""The grounded-answer contract shared by all providers.

A model answer is a list of *claims*. Each claim is a single statement and must
carry one or more pieces of *evidence*: the id of a context chunk and a verbatim
quote from it. Nothing in this structure is trusted until the verifier has
checked every quote against the stored source text.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator


class Evidence(BaseModel):
    chunk_id: str = Field(..., description="Context chunk id exactly as given, e.g. ML25002A104:p20:c1")
    quote: str = Field(..., description="Verbatim excerpt copied from that chunk")

    @field_validator("chunk_id", "quote")
    @classmethod
    def _strip(cls, v: str) -> str:
        return (v or "").strip()


class Claim(BaseModel):
    text: str = Field(..., description="One statement answering (part of) the question")
    evidence: list[Evidence] = Field(default_factory=list)
    kind: Literal["statement", "connective"] = "statement"


class ModelAnswer(BaseModel):
    status: Literal["answered", "partial", "not_found"] = "answered"
    claims: list[Claim] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SupportVerdict(BaseModel):
    verdict: Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"]
    reason: str = ""


ANSWER_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "claims", "notes"],
    "properties": {
        "status": {"type": "string", "enum": ["answered", "partial", "not_found"]},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "evidence"],
                "properties": {
                    "text": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["chunk_id", "quote"],
                            "properties": {"chunk_id": {"type": "string"}, "quote": {"type": "string"}},
                        },
                    },
                },
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
}

VERDICT_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "reason"],
    "properties": {
        "verdict": {"type": "string", "enum": ["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"]},
        "reason": {"type": "string"},
    },
}


class ParseError(ValueError):
    pass


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def extract_json_object(text: str) -> dict:
    """Parse the first JSON object in *text*, tolerating markdown fences and preambles."""
    if text is None:
        raise ParseError("empty model output")
    t = text.strip()
    m = _FENCE_RE.match(t)
    if m:
        t = m.group(1)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # find the outermost {...}
    start = t.find("{")
    if start < 0:
        raise ParseError("no JSON object in model output")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(t[start : i + 1])
                except json.JSONDecodeError as exc:
                    raise ParseError(f"invalid JSON in model output: {exc}") from exc
                if isinstance(obj, dict):
                    return obj
                break
    raise ParseError("unbalanced JSON in model output")


def parse_model_answer(text: str) -> ModelAnswer:
    obj = extract_json_object(text)
    try:
        return ModelAnswer.model_validate(obj)
    except ValidationError as exc:
        raise ParseError(f"model output does not match the answer schema: {exc}") from exc


def parse_verdict(text: str) -> SupportVerdict:
    obj = extract_json_object(text)
    v = str(obj.get("verdict", "")).upper().replace(" ", "_")
    if v not in ("SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"):
        raise ParseError(f"unknown verdict {v!r}")
    return SupportVerdict(verdict=v, reason=str(obj.get("reason", "")))
