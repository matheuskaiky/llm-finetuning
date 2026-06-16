"""Built-in guardrails: PII masking, jailbreak blocking, unsafe-topic blocking.

Each is registered in ``GUARDRAILS`` and can be composed by a ``GuardrailLayer``.
Heuristic/regex based (no ML stack); a model-based classifier can be added later as
another registered guardrail without touching the layer (OCP).
"""

from __future__ import annotations

import re

from . import GUARDRAILS
from .core import Guardrail, GuardrailResult
from .pii import mask_pii

# Jailbreak / prompt-injection markers (case-insensitive substrings).
_JAILBREAK = (
    "ignore previous instructions", "ignore as instrucoes anteriores",
    "desconsidere as instrucoes", "you are dan", "modo desenvolvedor",
    "developer mode", "sem restricoes", "sem filtros", "jailbreak",
    "finja que", "aja como se nao tivesse regras",
)

# Markers of clearly unsafe requests (weapons, self-harm facilitation, etc.).
_UNSAFE = (
    "como fabricar uma bomba", "como fazer uma bomba", "fabricar explosivo",
    "comprar arma ilegal", "como invadir o sistema", "roubar senha",
    "fabricar drogas", "como me matar",
)


@GUARDRAILS.register("pii_mask")
class PiiMaskGuardrail(Guardrail):
    """Mask Brazilian PII. By default on the output (protect leaked data)."""

    name = "pii_mask"

    def __init__(self, stages: tuple[str, ...] = ("output",)) -> None:
        self.stages = stages

    def apply(self, text: str, stage: str) -> GuardrailResult:
        masked, n = mask_pii(text)
        reasons = [f"pii_mask:{n}"] if n else []
        return GuardrailResult(allowed=True, text=masked, reasons=reasons)


@GUARDRAILS.register("jailbreak_block")
class JailbreakGuardrail(Guardrail):
    """Block prompt-injection / jailbreak attempts on the input."""

    name = "jailbreak_block"
    stages = ("input",)
    _refusal = "Desculpe, nao posso atender a esse pedido."

    def apply(self, text: str, stage: str) -> GuardrailResult:
        low = text.casefold()
        if any(m in low for m in _JAILBREAK):
            return GuardrailResult(False, self._refusal, ["jailbreak_block"])
        return GuardrailResult(True, text, [])


@GUARDRAILS.register("unsafe_block")
class UnsafeTopicGuardrail(Guardrail):
    """Block clearly unsafe requests on input and unsafe content on output."""

    name = "unsafe_block"
    _refusal = "Desculpe, nao posso ajudar com esse pedido."

    def apply(self, text: str, stage: str) -> GuardrailResult:
        low = re.sub(r"\s+", " ", text.casefold())
        if any(m in low for m in _UNSAFE):
            return GuardrailResult(False, self._refusal, ["unsafe_block"])
        return GuardrailResult(True, text, [])
