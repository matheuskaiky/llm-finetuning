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


def word_shingles(text: str, k: int = 5) -> set[str]:
    """k-word shingles of a text (for near-duplicate detection)."""
    words = text.split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def near_dup_keep_mask(texts: list[str], threshold: float = 0.85, num_perm: int = 64) -> list[bool]:
    """Keep-mask that drops near-duplicate texts (MinHash/LSH), keeping the first of
    each near-duplicate cluster. Collapses repetitive licitacao chunks without
    losing the unique ones."""
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep: list[bool] = []
    for i, t in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        for sh in word_shingles(t):
            m.update(sh.encode("utf-8"))
        if lsh.query(m):  # a near-duplicate of something already kept
            keep.append(False)
        else:
            lsh.insert(str(i), m)
            keep.append(True)
    return keep


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
