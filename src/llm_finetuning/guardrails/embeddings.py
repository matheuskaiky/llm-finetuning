"""Guardrail semântico: bloqueia por similaridade de embedding, não por substring.

Generaliza para paráfrase, tradução e reformulação (role-play, "modo
desenvolvedor" etc.) que o `filters.py` baseado em substring não cobre.
"""

from __future__ import annotations

from functools import lru_cache

from . import GUARDRAILS
from .core import Guardrail, GuardrailResult
from .seeds import JAILBREAK_SEEDS, UNSAFE_SEEDS

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL_NAME)


@lru_cache(maxsize=1)
def _seed_embeddings():
    import numpy as np
    enc = _encoder()
    jb = enc.encode(list(JAILBREAK_SEEDS), normalize_embeddings=True)
    un = enc.encode(list(UNSAFE_SEEDS), normalize_embeddings=True)
    return np.asarray(jb), np.asarray(un)


def _max_similarity(text: str) -> tuple[float, float]:
    """Retorna (sim_max_jailbreak, sim_max_unsafe) para ``text``."""
    import numpy as np
    enc = _encoder()
    jb, un = _seed_embeddings()
    vec = enc.encode([text], normalize_embeddings=True)[0]
    return float(np.max(jb @ vec)), float(np.max(un @ vec))


@GUARDRAILS.register("semantic_block")
class SemanticGuardrail(Guardrail):
    """Bloqueia entradas semanticamente próximas de jailbreak/unsafe conhecidos."""

    name = "semantic_block"
    stages = ("input",)
    _refusal = "Desculpe, não posso atender a esse pedido."

    def __init__(self, threshold: float = 0.55) -> None:
        self.threshold = threshold

    def apply(self, text: str, stage: str) -> GuardrailResult:
        sim_jb, sim_un = _max_similarity(text)
        if sim_jb >= self.threshold:
            return GuardrailResult(False, self._refusal, [f"semantic_block:jailbreak:{sim_jb:.2f}"])
        if sim_un >= self.threshold:
            return GuardrailResult(False, self._refusal, [f"semantic_block:unsafe:{sim_un:.2f}"])
        return GuardrailResult(True, text, [])
