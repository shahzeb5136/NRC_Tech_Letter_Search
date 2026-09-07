"""OpenAI-compatible chat-completions provider.

Works with OpenAI, Core42 (``OPENAI_BASE_URL=https://api.core42.ai/v1``), Azure-style
gateways and local servers (vLLM, Ollama, LM Studio) that expose the
``/v1/chat/completions`` interface. Output is strict JSON that the verifier checks.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from openai import OpenAI

from nrc_rag.config import Settings
from nrc_rag.llm.base import ContextItem, Generation, LLMProvider, b64, context_block_text
from nrc_rag.llm.prompts import (
    FIGURE_DESCRIPTION_SYSTEM,
    GROUNDED_SYSTEM,
    SUPPORT_JUDGE_SYSTEM,
    figure_description_prompt,
    grounded_user_prompt,
    support_judge_prompt,
)
from nrc_rag.llm.schema import SupportVerdict, parse_model_answer, parse_verdict

log = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.openai_model
        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url or None, timeout=settings.llm_timeout_seconds, max_retries=2)
        self._json_mode_ok: Optional[bool] = None  # learned on first call

    def _chat(self, messages: list[dict], max_tokens: int, json_mode: bool = True) -> tuple[str, dict, str]:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        if json_mode and self._json_mode_ok is not False:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
            if json_mode and self._json_mode_ok is None:
                self._json_mode_ok = True
        except Exception as exc:
            # Some gateways reject response_format / temperature; retry without them once.
            if "response_format" in kwargs:
                log.info("gateway rejected response_format (%s); retrying without it", exc)
                self._json_mode_ok = False
                kwargs.pop("response_format", None)
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise
        choice = resp.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise RuntimeError("model output was truncated (max_tokens); retry with a narrower question")
        text = choice.message.content or ""
        usage = {}
        if getattr(resp, "usage", None):
            usage = {
                "input_tokens": getattr(resp.usage, "prompt_tokens", None),
                "output_tokens": getattr(resp.usage, "completion_tokens", None),
            }
        return text, usage, getattr(resp, "model", self.model) or self.model

    def generate_grounded(self, question: str, items: list[ContextItem]) -> Generation:
        user_content: list[dict] = []
        for it in items:
            if it.image_png:
                user_content.append({"type": "text", "text": f"Image attached for figure chunk {it.chunk_id}:"})
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(it.image_png)}"}})
        user_content.append({"type": "text", "text": grounded_user_prompt(question, context_block_text(items))})
        messages = [
            {"role": "system", "content": GROUNDED_SYSTEM},
            {"role": "user", "content": user_content if len(user_content) > 1 else user_content[0]["text"]},
        ]
        text, usage, model = self._chat(messages, max_tokens=4000)
        answer = parse_model_answer(text)
        return Generation(answer=answer, raw_text=text, provider=self.name, model=model, usage=usage, request_meta={"base_url": self.settings.openai_base_url or "https://api.openai.com/v1", "temperature": 0}, mode="json")

    def judge_support(self, claim_text: str, quotes: list[str]) -> SupportVerdict:
        messages = [
            {"role": "system", "content": SUPPORT_JUDGE_SYSTEM},
            {"role": "user", "content": support_judge_prompt(claim_text, quotes)},
        ]
        text, _, _ = self._chat(messages, max_tokens=400)
        return parse_verdict(text)

    def describe_figure(self, png: bytes, caption: str, nearby_text: str) -> str:
        messages = [
            {"role": "system", "content": FIGURE_DESCRIPTION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(png)}"}},
                    {"type": "text", "text": figure_description_prompt(caption, nearby_text)},
                ],
            },
        ]
        text, _, _ = self._chat(messages, max_tokens=600, json_mode=False)
        return text.strip()
