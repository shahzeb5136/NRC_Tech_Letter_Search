"""Google Gemini provider (google-genai SDK). Strict JSON output, verified locally."""

from __future__ import annotations

import logging

from google import genai
from google.genai import types

from nrc_rag.config import Settings
from nrc_rag.llm.base import ContextItem, Generation, LLMProvider, context_block_text
from nrc_rag.llm.prompts import (
    FIGURE_DESCRIPTION_SYSTEM,
    GROUNDED_SYSTEM,
    SUPPORT_JUDGE_SYSTEM,
    figure_description_prompt,
    grounded_user_prompt,
    support_judge_prompt,
)
from nrc_rag.llm.schema import ANSWER_JSON_SCHEMA, VERDICT_JSON_SCHEMA, SupportVerdict, parse_model_answer, parse_verdict

log = logging.getLogger(__name__)


def _strip_additional_properties(schema: dict) -> dict:
    """Gemini's schema subset does not accept additionalProperties."""
    if isinstance(schema, dict):
        return {k: _strip_additional_properties(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_strip_additional_properties(x) for x in schema]
    return schema


class GoogleProvider(LLMProvider):
    name = "google"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.google_model
        self.client = genai.Client(api_key=settings.google_api_key)

    def _generate(self, system: str, parts: list, max_tokens: int, schema: dict | None) -> tuple[str, dict]:
        # Gemini 2.5+ models spend "thinking" tokens out of max_output_tokens, so the budget must be generous.
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if schema else None,
            response_schema=_strip_additional_properties(schema) if schema else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        resp = self.client.models.generate_content(model=self.model, contents=parts, config=config)
        cand = resp.candidates[0] if getattr(resp, "candidates", None) else None
        finish = str(getattr(cand, "finish_reason", "") or "")
        text = resp.text or ""
        if not text.strip():
            if "MAX_TOKENS" in finish:
                raise RuntimeError("model output was truncated (max_output_tokens); retry with a narrower question")
            if "SAFETY" in finish or "BLOCK" in finish:
                raise RuntimeError(f"model declined the request ({finish})")
            raise RuntimeError(f"empty model output (finish_reason={finish or 'unknown'})")
        usage = {}
        um = getattr(resp, "usage_metadata", None)
        if um:
            usage = {"input_tokens": getattr(um, "prompt_token_count", None), "output_tokens": getattr(um, "candidates_token_count", None)}
        return text, usage

    def generate_grounded(self, question: str, items: list[ContextItem]) -> Generation:
        parts: list = []
        for it in items:
            if it.image_png:
                parts.append(f"Image attached for figure chunk {it.chunk_id}:")
                parts.append(types.Part.from_bytes(data=it.image_png, mime_type="image/png"))
        parts.append(grounded_user_prompt(question, context_block_text(items)))
        text, usage = self._generate(GROUNDED_SYSTEM, parts, 16000, ANSWER_JSON_SCHEMA)
        answer = parse_model_answer(text)
        return Generation(answer=answer, raw_text=text, provider=self.name, model=self.model, usage=usage, request_meta={"temperature": 0}, mode="json")

    def judge_support(self, claim_text: str, quotes: list[str]) -> SupportVerdict:
        text, _ = self._generate(SUPPORT_JUDGE_SYSTEM, [support_judge_prompt(claim_text, quotes)], 4000, VERDICT_JSON_SCHEMA)
        return parse_verdict(text)

    def describe_figure(self, png: bytes, caption: str, nearby_text: str) -> str:
        parts = [types.Part.from_bytes(data=png, mime_type="image/png"), figure_description_prompt(caption, nearby_text)]
        text, _ = self._generate(FIGURE_DESCRIPTION_SYSTEM, parts, 4000, None)
        return text.strip()
