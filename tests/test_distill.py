"""Unit tests for the logit-KD loss (Q4). Needs torch; skipped if absent."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from llm_finetuning.training.distill import kd_loss  # noqa: E402


def test_kd_loss_zero_when_teacher_equals_student():
    logits = torch.randn(2, 5, 7)
    labels = torch.full((2, 5), -100)
    labels[:, 2:] = 1  # some response tokens
    loss = kd_loss(logits, logits.clone(), labels, temperature=2.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_kd_loss_positive_when_distributions_differ():
    s = torch.zeros(1, 4, 6)
    t = torch.zeros(1, 4, 6)
    t[..., 0] = 10.0  # teacher peaks on token 0
    labels = torch.tensor([[-100, 1, 1, 1]])
    assert float(kd_loss(s, t, labels, temperature=2.0)) > 0.0


def test_kd_loss_ignores_fully_masked():
    s = torch.randn(1, 3, 5)
    t = torch.randn(1, 3, 5)
    labels = torch.full((1, 3), -100)
    # all prompt -> denom clamped to 1, masked sum is 0 -> loss 0
    assert float(kd_loss(s, t, labels, temperature=1.0)) == pytest.approx(0.0, abs=1e-6)


def test_logit_kd_registered():
    from llm_finetuning.core.registry import TRAINERS

    assert "logit_kd" in TRAINERS
