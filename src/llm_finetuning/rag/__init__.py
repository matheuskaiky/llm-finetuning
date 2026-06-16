"""Q5 RAG: agentic GraphRAG over the diarios corpus.

Combines a vector store (FAISS) with a knowledge graph (NetworkX) and a
self-reflective LangGraph agent. Heavy dependencies (faiss, sentence-transformers,
langgraph, transformers) are imported lazily inside the modules that use them, so
the pure logic (chunking, graph traversal, prompt/parse helpers) is importable and
unit-testable without the ML stack.
"""

from __future__ import annotations

from .config import RagConfig, load_rag_config

__all__ = ["RagConfig", "load_rag_config"]
