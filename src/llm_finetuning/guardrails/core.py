"""Guardrail abstractions: a result, the strategy base, and the composing layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GuardrailResult:
    """Outcome of applying a guardrail (or the whole layer) to a text.

    ``allowed`` is False when the text must be blocked; ``text`` is the (possibly
    rewritten/masked) text to use; ``reasons`` lists which guardrails fired.
    """

    allowed: bool
    text: str
    reasons: list[str] = field(default_factory=list)


class Guardrail(ABC):
    """A single safety strategy applied at the ``input`` and/or ``output`` stage."""

    name: str = "guardrail"
    stages: tuple[str, ...] = ("input", "output")

    @abstractmethod
    def apply(self, text: str, stage: str) -> GuardrailResult:
        """Inspect/transform ``text`` for ``stage`` ('input' or 'output')."""


class GuardrailLayer:
    """Chains guardrails. Blocking stops the chain; masking/rewrites accumulate."""

    def __init__(self, guardrails: list[Guardrail]) -> None:
        self.guardrails = guardrails

    def _run(self, text: str, stage: str) -> GuardrailResult:
        reasons: list[str] = []
        for g in self.guardrails:
            if stage not in g.stages:
                continue
            res = g.apply(text, stage)
            reasons.extend(res.reasons)
            text = res.text
            if not res.allowed:
                return GuardrailResult(allowed=False, text=text, reasons=reasons)
        return GuardrailResult(allowed=True, text=text, reasons=reasons)

    def process_input(self, text: str) -> GuardrailResult:
        return self._run(text, "input")

    def process_output(self, text: str) -> GuardrailResult:
        return self._run(text, "output")
