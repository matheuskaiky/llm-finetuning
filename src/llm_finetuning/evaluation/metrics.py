"""Intrinsic next-token metrics: cross-entropy, perplexity and token accuracy.

The metrics operate on NumPy arrays of aligned ``logits`` and ``labels``; the
caller applies the causal shift. Tokens equal to ``ignore_index`` (default
``-100``) are skipped.

Definitions (natural log / nats):
    cross_entropy  = mean over tokens of  -log p(label)
    perplexity     = exp(cross_entropy)
    token_accuracy = fraction of tokens whose argmax prediction equals the label
"""

from __future__ import annotations

import numpy as np

from ..core.interfaces import Metric
from ..core.registry import METRICS


def _to_2d(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ``(..., vocab)`` logits and ``(...)`` labels to ``(N, vocab)``/``(N,)``."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    logits = logits.reshape(-1, logits.shape[-1])
    labels = labels.reshape(-1)
    return logits, labels


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


@METRICS.register("cross_entropy")
class CrossEntropy(Metric):
    """Mean token-level negative log-likelihood (in nats)."""

    name = "cross_entropy"

    def __init__(self, ignore_index: int = -100) -> None:
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self._nll_sum = 0.0
        self._n_tokens = 0

    def update(self, logits: np.ndarray, labels: np.ndarray) -> None:
        logits, labels = _to_2d(logits, labels)
        mask = labels != self.ignore_index
        if not mask.any():
            return
        logits, labels = logits[mask], labels[mask]
        log_probs = _log_softmax(logits)
        nll = -log_probs[np.arange(labels.shape[0]), labels]
        self._nll_sum += float(nll.sum())
        self._n_tokens += int(labels.shape[0])

    def compute(self) -> float:
        if self._n_tokens == 0:
            return float("nan")
        return self._nll_sum / self._n_tokens


@METRICS.register("perplexity")
class Perplexity(CrossEntropy):
    """Exponentiated cross-entropy."""

    name = "perplexity"

    def compute(self) -> float:
        ce = super().compute()
        return float(np.exp(ce))


@METRICS.register("token_accuracy")
class TokenAccuracy(Metric):
    """Fraction of next tokens predicted correctly (greedy argmax)."""

    name = "token_accuracy"

    def __init__(self, ignore_index: int = -100) -> None:
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self._correct = 0
        self._n_tokens = 0

    def update(self, logits: np.ndarray, labels: np.ndarray) -> None:
        logits, labels = _to_2d(logits, labels)
        mask = labels != self.ignore_index
        if not mask.any():
            return
        preds = logits[mask].argmax(axis=-1)
        self._correct += int((preds == labels[mask]).sum())
        self._n_tokens += int(mask.sum())

    def compute(self) -> float:
        if self._n_tokens == 0:
            return float("nan")
        return self._correct / self._n_tokens
