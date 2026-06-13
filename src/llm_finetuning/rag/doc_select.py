"""Document selection helpers: classify and balance gazette documents.

Procurement (licitacao) notices have a very repetitive structure and tend to
dominate the gazette corpus, which can swamp the vector store and the knowledge
graph. These helpers detect licitacao-heavy documents and downsample them so they
are paired in count with the other document types. Pure logic (no ML stack).
"""

from __future__ import annotations

import random

# Markers that signal a procurement/licitacao document.
LICITACAO_MARKERS = (
    "licitac",
    "licitaç",
    "pregao",
    "pregão",
    "tomada de prec",
    "tomada de preç",
    "concorrenc",
    "concorrênc",
    "edital",
    "homologac",
    "homologaç",
    "adjudicac",
    "adjudicaç",
    "dispensa de licit",
    "inexigibilidade",
    "registro de prec",
    "registro de preç",
    "processo licitat",
    "menor preco",
    "menor preço",
    "ata de registro",
)


def licitacao_score(text: str) -> int:
    """Number of distinct licitacao markers present (case-insensitive)."""
    t = text.casefold()
    return sum(1 for m in LICITACAO_MARKERS if m in t)


def is_licitacao(text: str, min_hits: int = 2) -> bool:
    """A document is licitacao-type when it hits at least ``min_hits`` markers."""
    return licitacao_score(text) >= min_hits


def balance_by_licitacao(
    docs: list[tuple[str, str]],
    seed: int = 42,
    max_total: int | None = None,
    min_hits: int = 2,
) -> list[tuple[str, str]]:
    """Downsample licitacao docs so they are paired with the non-licitacao count.

    ``docs`` is a list of ``(doc_id, text)``. Returns a shuffled list with equal
    numbers of licitacao and non-licitacao documents (``2*k``), where ``k`` is the
    smaller class size (further capped by ``max_total//2`` when given). Selection is
    deterministic for a fixed ``seed``.
    """
    lic = [d for d in docs if is_licitacao(d[1], min_hits)]
    oth = [d for d in docs if not is_licitacao(d[1], min_hits)]
    k = min(len(lic), len(oth))
    if max_total is not None:
        k = min(k, max_total // 2)
    rng = random.Random(seed)
    selected = rng.sample(lic, k) + rng.sample(oth, k)
    rng.shuffle(selected)
    return selected
