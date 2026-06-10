"""Tests for the text corpus loader and token-block chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_finetuning.data.text_corpus import TextCorpusLoader, chunk_token_ids


def test_chunk_drops_remainder_by_default() -> None:
    blocks = chunk_token_ids(list(range(10)), block_size=4)
    assert blocks == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_chunk_keeps_remainder_when_requested() -> None:
    blocks = chunk_token_ids(list(range(10)), block_size=4, drop_remainder=False)
    assert blocks == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_chunk_rejects_non_positive_block_size() -> None:
    with pytest.raises(ValueError):
        chunk_token_ids([1, 2, 3], block_size=0)


def test_text_corpus_loader_reads_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("primeiro documento", encoding="utf-8")
    (tmp_path / "b.txt").write_text("segundo documento", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("nao deve ser lido", encoding="utf-8")

    docs = TextCorpusLoader(tmp_path).load()

    assert docs == ["primeiro documento", "segundo documento"]
