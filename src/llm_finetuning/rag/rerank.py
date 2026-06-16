"""Maximal Marginal Relevance (MMR) reranking (pure numpy).

MMR re-orders candidate vectors to balance relevance to the query against
redundancy among the picks, so the top-k is not several near-identical chunks (the
failure mode caused by repetitive licitacao documents). Kept separate and pure so
it is testable without faiss.
"""

from __future__ import annotations

from typing import Any


def mmr_select(query_vec: Any, cand_vecs: Any, k: int, lambda_: float = 0.5) -> list[int]:
    """Greedy MMR selection over candidate vectors.

    Args:
        query_vec: shape (d,), normalized.
        cand_vecs: shape (n, d), normalized.
        k: number of items to select.
        lambda_: 1.0 = pure relevance, 0.0 = pure diversity.

    Returns the selected candidate indices (into ``cand_vecs``), in MMR order.
    """
    import numpy as np

    n = cand_vecs.shape[0]
    k = min(k, n)
    if k <= 0:
        return []
    sim_q = cand_vecs @ query_vec  # relevance to the query
    sim_cc = cand_vecs @ cand_vecs.T  # pairwise candidate similarity
    selected: list[int] = []
    remaining = set(range(n))
    while len(selected) < k:
        best_i, best_score = None, -np.inf
        for i in remaining:
            redundancy = max((sim_cc[i, j] for j in selected), default=0.0)
            score = lambda_ * sim_q[i] - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        remaining.discard(best_i)
    return selected
