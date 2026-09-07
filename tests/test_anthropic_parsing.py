"""Tests for turning Claude citation blocks into claims (no network: the API call is stubbed)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")

from nrc_rag.config import Settings  # noqa: E402
from nrc_rag.llm.anthropic_provider import AnthropicProvider  # noqa: E402
from nrc_rag.llm.base import ContextItem  # noqa: E402


def _items():
    return [
        ContextItem(chunk_id="D1:p3:c1", doc_id="D1", page_number=3, kind="text", section="1.5", doc_title="Doc 1", tlr_number="TLR-1", text="The off-gas system removes xenon and krypton."),
        ContextItem(chunk_id="D2:p8:f1", doc_id="D2", page_number=8, kind="figure", section="2", doc_title="Doc 2", tlr_number="TLR-2", text="Figure 3. Iodine sorbent loading versus temperature.", figure_id="D2:p8:f1"),
    ]


def _msg(blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, model="claude-opus-5", usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0), stop_details=None, _request_id="req_1")


def _provider(msg):
    p = AnthropicProvider(Settings(anthropic_api_key="test-key"))
    p._create = lambda **kwargs: msg  # type: ignore[method-assign]
    return p


def _cit(idx, text):
    return SimpleNamespace(type="char_location", document_index=idx, cited_text=text, document_title="", start_char_index=0, end_char_index=len(text))


def test_cited_blocks_become_claims_with_chunk_ids():
    blocks = [
        SimpleNamespace(type="text", text="The report states that ", citations=None),
        SimpleNamespace(type="text", text="the off-gas system removes xenon and krypton.", citations=[_cit(0, "removes xenon and krypton")]),
        SimpleNamespace(type="text", text=" From Figure: sorbent loading falls with temperature.", citations=[_cit(1, "Iodine sorbent loading versus temperature")]),
    ]
    gen = _provider(_msg(blocks)).generate_grounded("q", _items())
    assert gen.mode == "citations" and gen.answer.status == "answered"
    kinds = [c.kind for c in gen.answer.claims]
    assert kinds == ["connective", "statement", "statement"]
    assert gen.answer.claims[1].evidence[0].chunk_id == "D1:p3:c1"
    assert gen.answer.claims[1].evidence[0].quote == "removes xenon and krypton"
    assert gen.answer.claims[2].evidence[0].chunk_id == "D2:p8:f1"
    assert gen.usage["input_tokens"] == 10


def test_not_found_reply():
    gen = _provider(_msg([SimpleNamespace(type="text", text="NOT_FOUND", citations=None)])).generate_grounded("q", _items())
    assert gen.answer.status == "not_found" and gen.answer.claims == []


def test_uncited_substantive_text_is_a_statement_without_evidence():
    blocks = [SimpleNamespace(type="text", text="The system operates at 950 degrees Celsius for ten years without maintenance.", citations=None)]
    gen = _provider(_msg(blocks)).generate_grounded("q", _items())
    # no cited claims at all -> not_found, and the uncited sentence is dropped rather than shown
    assert gen.answer.status == "not_found"


def test_refusal_raises():
    with pytest.raises(RuntimeError):
        _provider(_msg([], stop_reason="refusal")).generate_grounded("q", _items())
