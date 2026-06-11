"""Tests for the text-normalization helpers of the PDF loader."""

from __future__ import annotations

from llm_finetuning.data.pdf_loader import _drop_repeated_lines, normalize_text


def test_normalize_collapses_whitespace_and_blank_lines() -> None:
    raw = "Linha   um  \n\n\n\nLinha\tdois   "
    assert normalize_text(raw) == "Linha um\n\nLinha dois"


def test_drop_repeated_lines_removes_headers() -> None:
    header = "DIARIO OFICIAL"
    pages = [
        f"{header}\nconteudo A",
        f"{header}\nconteudo B",
        f"{header}\nconteudo C",
    ]
    cleaned = _drop_repeated_lines(pages, threshold=0.6)
    assert all(header not in page for page in cleaned)
    assert "conteudo A" in cleaned[0]


def test_drop_repeated_lines_noop_for_few_pages() -> None:
    pages = ["a", "b"]
    assert _drop_repeated_lines(pages, threshold=0.6) == pages
