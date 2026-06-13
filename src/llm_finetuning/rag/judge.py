"""LLM-as-judge scoring for the RAG benchmark (0-5).

Scores an answer against a reference for accuracy and completeness. Prompt building
and score parsing are pure; the LLM call is a thin wrapper.
"""

from __future__ import annotations

import re
from typing import Any

JUDGE_SYSTEM = (
    "Voce avalia respostas comparando com uma resposta de referencia. Atribua uma "
    "nota inteira de 0 a 5 para precisao e completude (0 = errada/irrelevante, "
    "5 = correta e completa). Responda APENAS com o numero."
)


def build_judge_messages(question: str, expected: str, answer: str) -> list[dict[str, str]]:
    user = (
        f"Pergunta: {question}\n\nResposta de referencia: {expected}\n\n"
        f"Resposta avaliada: {answer}\n\nNota (0 a 5):"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_score(raw: str) -> int:
    """Extract the first integer 0-5 from the judge output, clamped (pure)."""
    m = re.search(r"-?\d+", raw)
    if not m:
        return 0
    return max(0, min(5, int(m.group())))


def llm_judge(llm: Any, question: str, expected: str, answer: str) -> int:
    """Score one (answer vs reference) pair via the LLM."""
    raw = llm.chat(build_judge_messages(question, expected, answer), max_new_tokens=8)
    return parse_score(raw)
