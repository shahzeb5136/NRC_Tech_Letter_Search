"""Unit tests for the deterministic quote verifier."""

from __future__ import annotations

from nrc_rag.index.store import ChunkRow, PageRow
from nrc_rag.verify.quote_verifier import best_window_similarity, verify_quote

SOURCE = (
    "The off-gas system removes fission product gases, primarily xenon and krypton, "
    "from the cover gas. Operating procedures may be established to account for routine "
    "maintenance of off-gas system sub-\ncomponents and to “minimize” releases – within limits."
)


def make_chunk(text: str = SOURCE, kind: str = "text") -> ChunkRow:
    return ChunkRow(
        chunk_id="DOC1:p3:c1", doc_id="DOC1", page_number=3, kind=kind, section="1.5", section_path="1 > 1.5",
        text=text, char_start=0, char_end=len(text), bbox=[72, 70, 540, 200], figure_id=None, token_count=60,
    )


def make_page(text: str) -> PageRow:
    return PageRow(doc_id="DOC1", page_number=3, label="", width=612, height=792, text=text, section="", section_path="", is_toc=False, blocks=[])


def test_exact_match_in_chunk():
    c = verify_quote("removes fission product gases, primarily xenon and krypton", make_chunk(), None)
    assert c.status == "exact" and c.location == "chunk" and c.score == 1.0


def test_typographic_differences_are_exact():
    # straight quotes / hyphen instead of curly quotes / en dash, different whitespace and case
    c = verify_quote('to "MINIMIZE" releases - within   limits', make_chunk(), None)
    assert c.status == "exact"


def test_hyphenation_across_line_break():
    c = verify_quote("off-gas system subcomponents and to", make_chunk(), None)
    assert c.status == "exact"


def test_fabricated_quote_fails():
    c = verify_quote("the system operates at 950 degrees Celsius for ten years", make_chunk(), None)
    assert c.status == "failed"
    assert c.score < 0.92


def test_near_verbatim_is_fuzzy_not_exact():
    # one dropped word -> fuzzy (still counts as evidence but flagged)
    c = verify_quote("Operating procedures may be established to account for maintenance of off-gas system", make_chunk(), None)
    assert c.status in ("fuzzy", "failed")
    if c.status == "fuzzy":
        assert 0.92 <= c.score < 1.0


def test_page_level_match_outside_chunk():
    page = make_page(SOURCE + " Additional sentence only present on the page text.")
    c = verify_quote("Additional sentence only present on the page text", make_chunk(), page)
    assert c.status == "exact" and c.location == "page"


def test_too_short_quote_rejected():
    c = verify_quote("xenon and", make_chunk(), None)
    assert c.status == "failed" and "short" in c.reason


def test_unknown_chunk_rejected():
    c = verify_quote("removes fission product gases", None, None)
    assert c.status == "failed" and "not part of the retrieved context" in c.reason


def test_table_markdown_pipes_ignored():
    table = "Table on page 9\n\n|Overheating|Increase in temperature, decreased decontamination factor|\n|---|---|"
    c = verify_quote("Overheating Increase in temperature, decreased decontamination factor", make_chunk(table, kind="table"), None)
    assert c.status == "exact"


def test_best_window_similarity_bounds():
    r, w = best_window_similarity("abcdefghijklmnopqrstuvwxyz" * 3, "hijklmnop")
    assert 0.0 <= r <= 1.0 and w


def test_figure_quote_found_in_page_text_is_flagged():
    fig_text = "Figure on page 3 of DOC1: Figure 1-1. Schematic of the off-gas system.\n\nContext: The off-gas system removes fission product gases."
    chunk = make_chunk(fig_text, kind="figure")
    page = make_page(SOURCE + " Figure 1-1. Schematic of the off-gas system.")
    c = verify_quote("Figure 1-1. Schematic of the off-gas system", chunk, page)
    assert c.status == "exact" and c.location == "chunk" and c.in_page_text is True
    # a quote that only exists in an AI description is not page text
    chunk2 = make_chunk(fig_text + "\n\nDescription (AI-generated): the plot peaks at 600 C", kind="figure")
    c2 = verify_quote("the plot peaks at 600 C", chunk2, page)
    assert c2.status == "exact" and c2.in_page_text is False
