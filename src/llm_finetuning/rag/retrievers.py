"""Retrievers: vector (semantic) and graph (relational), behind one interface.

A retriever turns a question into a context string for the generator. The vector
retriever does semantic search over chunks; the graph retriever resolves entities
mentioned in the question and renders the surrounding relations.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ÿ]{4,}")
_STOP = {
    "qual",
    "quais",
    "quem",
    "onde",
    "quando",
    "como",
    "quanto",
    "para",
    "pela",
    "pelo",
    "sobre",
    "entre",
    "dos",
    "das",
    "com",
    "que",
    "uma",
    "uns",
    "essa",
    "esse",
    "foram",
    "tem",
    "teve",
}


def candidate_entity_terms(question: str) -> list[str]:
    """Heuristic terms from a question to look up in the graph (pure, testable)."""
    terms = [w for w in _WORD_RE.findall(question) if w.casefold() not in _STOP]
    # Keep order, drop duplicates (case-insensitive).
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        key = t.casefold()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


class Retriever(Protocol):
    name: str

    def retrieve(self, question: str) -> str: ...


class VectorRetriever:
    name = "vector"

    def __init__(self, store: Any, top_k: int = 5) -> None:
        self.store = store
        self.top_k = top_k

    def retrieve(self, question: str) -> str:
        hits = self.store.search(question, k=self.top_k)
        return "\n\n".join(f"[{h.doc_id}] {h.text}" for h in hits)


class GraphRetriever:
    name = "graph"

    def __init__(self, graph: Any, hops: int = 2) -> None:
        self.graph = graph
        self.hops = hops

    def retrieve(self, question: str) -> str:
        terms = candidate_entity_terms(question)
        return self.graph.context_for_entities(terms, hops=self.hops)
