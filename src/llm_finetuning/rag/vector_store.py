"""FAISS vector store over text chunks (lazy faiss / sentence-transformers).

Stores normalized embeddings in an inner-product index (cosine similarity) plus a
parallel list of chunk texts and document ids for citations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Retrieved:
    text: str
    doc_id: str
    score: float


class Embedder:
    """sentence-transformers wrapper producing normalized embeddings."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: list[str]) -> Any:
        import numpy as np

        self._ensure_loaded()
        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype="float32")

    def unload(self) -> None:
        """Release the embedder's GPU memory (call before loading a large LLM so
        device_map="auto" does not offload model layers to CPU)."""
        if self._model is None:
            return
        import gc

        del self._model
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class VectorStore:
    """In-memory FAISS index with persisted texts/metadata."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._index: Any = None
        self._texts: list[str] = []
        self._doc_ids: list[str] = []

    def build(self, texts: list[str], doc_ids: list[str]) -> None:
        import faiss

        if len(texts) != len(doc_ids):
            raise ValueError("texts and doc_ids must have the same length")
        vectors = self.embedder.encode(texts)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._index = index
        self._texts = list(texts)
        self._doc_ids = list(doc_ids)

    def search(self, query: str, k: int = 5) -> list[Retrieved]:
        if self._index is None:
            raise RuntimeError("index not built/loaded")
        vec = self.embedder.encode([query])
        scores, idxs = self._index.search(vec, min(k, len(self._texts)))
        out: list[Retrieved] = []
        for score, i in zip(scores[0], idxs[0], strict=False):
            if i < 0:
                continue
            out.append(Retrieved(self._texts[i], self._doc_ids[i], float(score)))
        return out

    def save(self, directory: str | Path) -> Path:
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        (directory / "meta.json").write_text(
            json.dumps({"texts": self._texts, "doc_ids": self._doc_ids}, ensure_ascii=False),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder) -> VectorStore:
        import faiss

        directory = Path(directory)
        store = cls(embedder)
        store._index = faiss.read_index(str(directory / "index.faiss"))
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        store._texts = meta["texts"]
        store._doc_ids = meta["doc_ids"]
        return store
