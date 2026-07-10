"""Guardrails (Q6): composable input/output safety filters over a model/RAG.

A ``GuardrailLayer`` chains ``Guardrail`` strategies (registered in ``GUARDRAILS``)
that can mask sensitive data, block unsafe inputs/outputs or rewrite them. Importing
this package registers the built-in guardrails. Pure logic (regex/heuristics); no ML
stack, so the protection layer is unit-testable in isolation.
"""

from __future__ import annotations

from ..core.registry import Registry
from .core import Guardrail, GuardrailLayer, GuardrailResult

GUARDRAILS: Registry[Guardrail] = Registry("guardrail")

from . import filters as _filters  # noqa: E402,F401  (registers built-ins)
from . import embeddings as _embeddings  # noqa: E402,F401  (registra semantic_block)
from .pii import mask_pii  # noqa: E402

__all__ = [
    "GUARDRAILS",
    "Guardrail",
    "GuardrailLayer",
    "GuardrailResult",
    "mask_pii",
]
