"""Tests for the intrinsic metrics with known closed-form values."""

from __future__ import annotations

import math

import numpy as np

from llm_finetuning.evaluation.metrics import CrossEntropy, Perplexity, TokenAccuracy


def test_uniform_logits_cross_entropy_and_perplexity() -> None:
    # Uniform logits over a vocab of 3 -> p = 1/3 for every class.
    logits = np.zeros((4, 3))
    labels = np.array([0, 1, 2, 0])

    ce = CrossEntropy()
    ce.update(logits, labels)
    assert math.isclose(ce.compute(), math.log(3), rel_tol=1e-9)

    ppl = Perplexity()
    ppl.update(logits, labels)
    assert math.isclose(ppl.compute(), 3.0, rel_tol=1e-9)


def test_ignore_index_is_skipped() -> None:
    logits = np.zeros((2, 3))
    labels = np.array([0, -100])

    ce = CrossEntropy()
    ce.update(logits, labels)
    # Only one valid token contributes.
    assert math.isclose(ce.compute(), math.log(3), rel_tol=1e-9)


def test_token_accuracy() -> None:
    # argmax of row 0 is class 0 (!= label 1); row 1 is class 0 (== label 0).
    logits = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    labels = np.array([1, 0])

    acc = TokenAccuracy()
    acc.update(logits, labels)
    assert math.isclose(acc.compute(), 0.5, rel_tol=1e-9)


def test_accumulation_across_updates() -> None:
    acc = TokenAccuracy()
    acc.update(np.array([[1.0, 0.0]]), np.array([0]))  # correct
    acc.update(np.array([[1.0, 0.0]]), np.array([1]))  # wrong
    assert math.isclose(acc.compute(), 0.5, rel_tol=1e-9)


def test_empty_metric_is_nan() -> None:
    assert math.isnan(CrossEntropy().compute())
    assert math.isnan(TokenAccuracy().compute())
