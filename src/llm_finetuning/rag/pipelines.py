"""RAG runners (Strategy): one class per RAG mode, selected by name.

Each runner answers a question with a different pipeline so the evaluator can treat
them uniformly. Adding a mode means adding a runner and registering it (OCP); the
evaluator does not change. ``standard`` is plain retrieve-then-generate; the
agentic runners wrap the self-reflective LangGraph agent (with or without the
knowledge graph).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .agent import build_agent, standard_rag


@dataclass
class RunResult:
    answer: str
    corrected: bool = False  # whether the self-reflective critic forced a retry


class RagRunner(Protocol):
    name: str

    def answer(self, question: str) -> RunResult: ...


class StandardRunner:
    """Plain RAG: vector retrieve once, generate once (no agent, no graph)."""

    name = "standard"

    def __init__(self, llm: Any, vector_retriever: Any, **_: Any) -> None:
        self.llm = llm
        self.vector_retriever = vector_retriever

    def answer(self, question: str) -> RunResult:
        return RunResult(standard_rag(self.llm, self.vector_retriever, question))


class AgenticRunner:
    """Self-reflective agent, with (`use_graph=True`) or without the graph."""

    def __init__(
        self,
        name: str,
        llm: Any,
        vector_retriever: Any,
        graph_retriever: Any,
        max_reflections: int = 2,
        use_graph: bool = True,
    ) -> None:
        self.name = name
        self._agent = build_agent(
            llm, vector_retriever, graph_retriever, max_reflections, use_graph=use_graph
        )

    def answer(self, question: str) -> RunResult:
        state = self._agent.invoke({"question": question})
        return RunResult(state.get("answer", ""), bool(state.get("corrected", False)))


# Mode name -> factory. Register a new mode here; the evaluator stays unchanged.
RUNNERS: dict[str, Callable[..., RagRunner]] = {
    "standard": lambda **kw: StandardRunner(**kw),
    "agentic_vector": lambda **kw: AgenticRunner(name="agentic_vector", use_graph=False, **kw),
    "agentic_graph": lambda **kw: AgenticRunner(name="agentic_graph", use_graph=True, **kw),
}


def build_runner(
    mode: str,
    llm: Any,
    vector_retriever: Any,
    graph_retriever: Any,
    max_reflections: int = 2,
) -> RagRunner:
    """Build the runner registered under ``mode``."""
    try:
        factory = RUNNERS[mode]
    except KeyError:
        raise KeyError(f"unknown RAG mode {mode!r}; available: {sorted(RUNNERS)}") from None
    return factory(
        llm=llm,
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
        max_reflections=max_reflections,
    )
