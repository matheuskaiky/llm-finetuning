"""Mapeia cada guardrail à categoria da política declarativa (docs/POLITICA_SEGURANCA.md).

Existir esse mapeamento é o que torna uma recusa auditável: em vez de "bloqueado",
o motivo aponta pra uma categoria de política com justificativa e ação definidas.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyCategory:
    code: str
    title: str
    action: str  # "mask" | "block" | "rewrite"
    basis: str    # marco regulatório ou racional de negócio


POLICY: dict[str, PolicyCategory] = {
    "pii_mask": PolicyCategory(
        code="P1", title="Dados Pessoais", action="mask",
        basis="LGPD Art. 5º",
    ),
    "jailbreak_block": PolicyCategory(
        code="P2", title="Manipulação de Instrução", action="block",
        basis="Robustez de sistema (Security)",
    ),
    "semantic_block": PolicyCategory(
        code="P2", title="Manipulação de Instrução", action="block",
        basis="Robustez de sistema (Security)",
    ),
    "unsafe_block": PolicyCategory(
        code="P3", title="Conteúdo Manifestamente Nocivo", action="block",
        basis="Segurança (Safety)",
    ),
}


def annotate(reason: str) -> str:
    """``\"jailbreak_block\"`` -> ``\"jailbreak_block [P2 Manipulação de Instrução]\"``."""
    key = reason.split(":", 1)[0]  # guardrails com sufixo, ex. "semantic_block:jailbreak:0.71"
    cat = POLICY.get(key)
    return f"{reason} [{cat.code} {cat.title}]" if cat else reason
