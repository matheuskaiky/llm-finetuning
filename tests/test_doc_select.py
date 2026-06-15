"""Unit tests for licitacao detection and balancing (pure, no ML stack)."""

from __future__ import annotations

from llm_finetuning.rag.doc_select import (
    balance_by_licitacao,
    downsample_licitacao,
    is_licitacao,
    licitacao_score,
)

_LIC = "Aviso de licitacao. Pregao eletronico. Edital. Homologacao do objeto."
_OTHER = "Portaria que nomeia servidor para o cargo de diretor da secretaria."


def test_licitacao_score_and_flag():
    assert licitacao_score(_LIC) >= 2
    assert is_licitacao(_LIC)
    assert licitacao_score(_OTHER) == 0
    assert not is_licitacao(_OTHER)


def test_single_marker_is_not_enough_by_default():
    assert not is_licitacao("menciona um edital apenas")  # 1 hit < min_hits=2


def test_balance_pairs_the_classes():
    docs = [(f"l{i}", _LIC) for i in range(20)] + [(f"o{i}", _OTHER) for i in range(5)]
    out = balance_by_licitacao(docs, seed=1)
    n_lic = sum(1 for _, t in out if is_licitacao(t))
    assert n_lic == len(out) - n_lic == 5  # paired down to the minority (5 each)


def test_balance_respects_max_total():
    docs = [(f"l{i}", _LIC) for i in range(50)] + [(f"o{i}", _OTHER) for i in range(50)]
    out = balance_by_licitacao(docs, seed=1, max_total=30)
    assert len(out) == 30
    n_lic = sum(1 for _, t in out if is_licitacao(t))
    assert n_lic == 15  # half each


def test_balance_deterministic_with_seed():
    docs = [(f"l{i}", _LIC) for i in range(20)] + [(f"o{i}", _OTHER) for i in range(20)]
    assert balance_by_licitacao(docs, seed=7) == balance_by_licitacao(docs, seed=7)


def test_downsample_only_touches_licitacao():
    docs = [(f"l{i}", _LIC) for i in range(20)] + [(f"o{i}", _OTHER) for i in range(7)]
    out = downsample_licitacao(docs, keep_fraction=0.5, seed=1)
    n_lic = sum(1 for _, t in out if is_licitacao(t))
    n_oth = len(out) - n_lic
    assert n_lic == 10  # half of the 20 licitacao docs kept
    assert n_oth == 7  # all non-licitacao docs preserved


def test_downsample_keep_fraction_bounds():
    docs = [(f"l{i}", _LIC) for i in range(10)] + [(f"o{i}", _OTHER) for i in range(4)]
    assert len(downsample_licitacao(docs, keep_fraction=0.0)) == 4  # all licitacao dropped
    assert len(downsample_licitacao(docs, keep_fraction=1.0)) == 14  # nothing dropped


def test_downsample_deterministic_with_seed():
    docs = [(f"l{i}", _LIC) for i in range(30)] + [(f"o{i}", _OTHER) for i in range(10)]
    assert downsample_licitacao(docs, seed=3) == downsample_licitacao(docs, seed=3)
