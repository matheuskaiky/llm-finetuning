"""Character-based text chunking for indexing (pure, no ML stack)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """A piece of a document, with provenance for retrieval citations."""

    doc_id: str
    index: int
    text: str


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Split ``text`` into overlapping windows of about ``chunk_size`` characters.

    Windows advance by ``chunk_size - overlap``. Each window is extended to the next
    whitespace so words are not cut mid-token; empty/blank windows are dropped.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    text = text.strip()
    if not text:
        return []
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # Extend to the next whitespace to avoid cutting a word, unless at the end.
        if end < n:
            nxt = text.find(" ", end)
            if nxt != -1 and nxt - end < 40:
                end = nxt
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start += step
    return chunks


def chunk_document(doc_id: str, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk one document into :class:`Chunk` records."""
    return [
        Chunk(doc_id=doc_id, index=i, text=t)
        for i, t in enumerate(chunk_text(text, chunk_size, overlap))
    ]
