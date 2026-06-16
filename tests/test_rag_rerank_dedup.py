"""Unit tests for MMR reranking and near-duplicate dedup (pure logic)."""

from __future__ import annotations

import numpy as np

from llm_finetuning.rag.doc_select import near_dup_keep_mask, word_shingles
from llm_finetuning.rag.rerank import mmr_select


def _unit(v):
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v)


def test_mmr_prefers_diverse_over_near_duplicate():
    query = _unit([1.0, 0.0, 0.0])
    cands = np.vstack([
        _unit([1.0, 0.0, 0.0]),   # 0: most relevant
        _unit([0.6, 0.8, 0.0]),   # 1: relevant
        _unit([0.6, 0.8, 0.0]),   # 2: near-duplicate of 1
        _unit([0.6, 0.0, 0.8]),   # 3: as relevant as 1 but diverse from it
    ])
    picks = mmr_select(query, cands, k=3, lambda_=0.5)
    assert picks[0] == 0  # most relevant first
    # the diverse one (3) is chosen over the near-duplicate (2)
    assert 2 not in picks
    assert set(picks) == {0, 1, 3}


def test_mmr_pure_relevance_when_lambda_one():
    query = _unit([1.0, 0.0, 0.0])
    cands = np.vstack([
        _unit([1.0, 0.0, 0.0]), _unit([0.6, 0.8, 0.0]),
        _unit([0.6, 0.8, 0.0]), _unit([0.6, 0.0, 0.8]),
    ])
    picks = mmr_select(query, cands, k=2, lambda_=1.0)
    assert picks[0] == 0  # diversity ignored; top relevance first


def test_mmr_handles_k_larger_than_n():
    q = _unit([1.0, 0.0])
    cands = np.vstack([_unit([1.0, 0.0]), _unit([0.0, 1.0])])
    assert len(mmr_select(q, cands, k=5)) == 2


def test_word_shingles():
    assert word_shingles("a b c", k=5) == {"a b c"}
    assert "a b" in word_shingles("a b c d", k=2)


def test_near_dup_keep_mask_drops_repetitive():
    base = "aviso de licitacao pregao eletronico edital homologacao objeto " * 5
    near = base + "extra"  # almost identical
    other = "portaria que nomeia servidor para o cargo de diretor da secretaria " * 5
    mask = near_dup_keep_mask([base, near, other], threshold=0.8)
    assert mask[0] is True  # first kept
    assert mask[1] is False  # near-duplicate dropped
    assert mask[2] is True  # distinct kept
