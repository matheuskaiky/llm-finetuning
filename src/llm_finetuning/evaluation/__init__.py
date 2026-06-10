"""Evaluation: intrinsic language-model metrics and the evaluator that runs them.

Importing this module registers the built-in metrics and evaluator.
"""

from __future__ import annotations

from .evaluator import LanguageModelEvaluator
from .metrics import CrossEntropy, Perplexity, TokenAccuracy

__all__ = [
    "LanguageModelEvaluator",
    "CrossEntropy",
    "Perplexity",
    "TokenAccuracy",
]
