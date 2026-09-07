"""Tests for model-output parsing and the chunker's provenance guarantees."""

from __future__ import annotations

import pytest

from nrc_rag.ingest.chunker import TokenCounter, chunk_page
from nrc_rag.ingest.pdf_extract import Block, PageData
from nrc_rag.llm.schema import ParseError, extract_json_object, parse_model_answer, parse_verdict


def test_parse_fenced_json():
    text = '```json\n{"status": "answered", "claims": [{"text": "A.", "evidence": [{"chunk_id": "D:p1:c1", "quote": "a quote"}]}], "notes": []}\n```'
    ans = parse_model_answer(text)
    assert ans.status == "answered" and ans.claims[0].evidence[0].chunk_id == "D:p1:c1"


def test_parse_with_preamble_and_nested_braces():
    text = 'Here is the answer: {"status": "partial", "claims": [], "notes": ["x {y} z"]} trailing'
    obj = extract_json_object(text)
    assert obj["status"] == "partial" and obj["notes"] == ["x {y} z"]


def test_parse_rejects_garbage():
    with pytest.raises(ParseError):
        parse_model_answer("no json here")


def test_parse_verdict_normalises():
    v = parse_verdict('{"verdict": "partially supported", "reason": "r"}')
    assert v.verdict == "PARTIALLY_SUPPORTED"


def _page(text_blocks: list[str]) -> PageData:
    blocks, parts, pos = [], [], 0
    for i, t in enumerate(text_blocks):
        if parts:
            pos += 2
        start = pos
        parts.append(t)
        pos += len(t)
        blocks.append(Block(72, 70 + 40 * i, 540, 100 + 40 * i, t, start, pos))
    return PageData(doc_id="DOC1", page_number=5, label="", width=612, height=792, text="\n\n".join(parts), blocks=blocks, section="2.1 Test", section_path="2 > 2.1 Test")


def test_chunks_are_verbatim_slices_with_bbox():
    sentences = [" ".join(f"Sentence {i} about off-gas systems and xenon removal." for i in range(k, k + 6)) for k in range(0, 60, 6)]
    page = _page(sentences)
    chunks = chunk_page(page, TokenCounter(), target_tokens=80, overlap_tokens=15)
    assert len(chunks) > 1
    for c in chunks:
        assert page.text[c.char_start:c.char_end] == c.text
        assert c.page_number == 5 and c.doc_id == "DOC1"
        assert c.bbox is not None and c.bbox[0] == 72
        assert c.chunk_id.startswith("DOC1:p5:c")
        assert c.section == "2.1 Test"
    # overlap: consecutive chunks share text
    assert chunks[1].char_start < chunks[0].char_end


def test_toc_pages_produce_no_chunks():
    page = _page(["1 Introduction ........ 3", "2 Methods ........ 7", "3 Results ........ 9"])
    page.is_toc = True
    assert chunk_page(page, TokenCounter()) == []
