"""Brazilian PII detection/masking (CPF, CNPJ, CEP, phone, email). Pure regex.

Relevant to the gazette/docente corpora, which carry names, ids and addresses of
public servants. ``mask_pii`` replaces matches with a typed placeholder.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("CPF", re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")),
    ("CNPJ", re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("CEP", re.compile(r"\b\d{5}-\d{3}\b")),
    ("TELEFONE", re.compile(r"\(?\b\d{2}\)?\s?9?\d{4}-\d{4}\b")),
]


def mask_pii(text: str) -> tuple[str, int]:
    """Replace Brazilian PII with ``[TIPO]`` placeholders. Returns (text, n_masked)."""
    n = 0
    for label, pat in _PATTERNS:
        text, k = pat.subn(f"[{label}]", text)
        n += k
    return text, n


def has_pii(text: str) -> bool:
    return any(pat.search(text) for _, pat in _PATTERNS)
