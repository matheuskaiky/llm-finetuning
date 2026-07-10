"""Self-reflective GraphRAG agent (SPAR) built on LangGraph.

Flow: Analyzer/Router (Plan) -> Retrieve (Act, vector and/or graph) -> Generate ->
Critic (Reflect). The critic checks groundedness and relevance; if it rejects and
the reflection budget is not exhausted, control loops back to the analyzer with the
critic feedback (widening retrieval). Routing and critic parsing are pure functions
tested without the LLM; langgraph is imported lazily.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict


class RagState(TypedDict, total=False):
    """Channels carried through the agent graph."""

    question: str
    route: str
    context: str
    answer: str
    approved: bool
    feedback: str
    reflections: int
    corrected: bool

_RELATIONAL_HINTS = (
    "relacao",
    "relação",
    "contrato",
    "licitacao",
    "licitação",
    "venceu",
    "ganhou",
    "entre",
    "quem tem",
    "quais empresas",
    "com a prefeitura",
    "fornecedor",
    "vinculo",
    "vínculo",
    "ligad",
)


def decide_route(question: str, force_both: bool = False) -> str:
    """Optional keyword-based router (not used by the default agent, which always
    consults the graph in graph mode). Kept as a pure, testable utility for a
    future cost-aware routing policy. Returns 'vector', 'graph' or 'both'."""
    if force_both:
        return "both"
    q = question.casefold()
    relational = any(h in q for h in _RELATIONAL_HINTS)
    return "both" if relational else "vector"


GENERATOR_SYSTEM = (
    "Você responde perguntas sobre diários oficiais de municípios usando SOMENTE o "
    "contexto fornecido. Se o contexto não contiver a resposta, diga que não há "
    "informação suficiente. Seja conciso e factual, em português."
)

CRITIC_SYSTEM = (
    "Você é um avaliador rigoroso. Dada a pergunta, o contexto e a resposta, "
    "verifique: (a) a resposta está 100 por cento embasada no contexto (sem "
    "alucinação)? (b) a resposta responde à pergunta? Responda APENAS com JSON: "
    '{"approved": true|false, "feedback": "o que falta ou corrigir"}.'
)


def build_generator_messages(question: str, context: str, feedback: str = "") -> list[dict[str, str]]:
    extra = f"\n\nObservação do avaliador anterior: {feedback}" if feedback else ""
    user = f"Contexto:\n{context or '(vazio)'}\n\nPergunta: {question}{extra}\n\nResposta:"
    return [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_critic_messages(question: str, context: str, answer: str) -> list[dict[str, str]]:
    user = (
        f"Pergunta: {question}\n\nContexto:\n{context or '(vazio)'}\n\n"
        f"Resposta a avaliar:\n{answer}\n\nJSON:"
    )
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_critic(raw: str) -> dict[str, Any]:
    """Parse the critic JSON into ``{'approved': bool, 'feedback': str}``.

    Defaults to approved=True on unparseable output so a flaky critic does not trap
    the loop; the reflection budget also bounds it.
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            return {
                "approved": bool(data.get("approved", True)),
                "feedback": str(data.get("feedback", "")).strip(),
            }
        except (json.JSONDecodeError, ValueError):
            pass
    # Fallback: look for an explicit boolean-ish token.
    approved = not re.search(r"\bfalse\b|\bnao\b|\bnão\b", raw.casefold())
    return {"approved": approved, "feedback": raw.strip()[:300]}


def standard_rag(llm: Any, vector_retriever: Any, question: str) -> str:
    """Plain (non-agentic) RAG: vector retrieve once, generate once. No graph, no
    critic loop. The baseline RAG to ablate against the agentic variants."""
    context = vector_retriever.retrieve(question)
    from llm_finetuning.guardrails import GUARDRAILS
    jb = GUARDRAILS.build("jailbreak_block")
    if context and not jb.apply(context, "input").allowed:
        context = ""
    return llm.chat(build_generator_messages(question, context))


def build_agent(
    llm: Any,
    vector_retriever: Any,
    graph_retriever: Any,
    max_reflections: int = 2,
    use_graph: bool = True,
) -> Any:
    """Compile the LangGraph self-reflective RAG agent.

    With ``use_graph=False`` the agent never routes to the knowledge graph (vector
    retrieval only), isolating the effect of the graph from the effect of the
    self-reflective loop. The returned object exposes ``invoke({"question": ...})``
    and yields a final state with ``answer``, ``route``, ``reflections`` and
    ``corrected`` (whether the critic forced at least one retry).
    """
    from langgraph.graph import END, StateGraph

    def analyze(state: dict[str, Any]) -> dict[str, Any]:
        # Always consult the graph in graph mode. The earlier keyword router
        # (decide_route) starved it: its hints were licitacao-centric, so non-bid
        # questions (nomeacoes, orgaos, pertencimento) never reached the rich graph
        # content. The graph context is still only appended when non-empty (see
        # retrieve), so this adds cost only when the graph has something to say.
        return {"route": "both" if use_graph else "vector"}

    def retrieve(state: dict[str, Any]) -> dict[str, Any]:
        route = state["route"]
        parts: list[str] = []
        if route in ("vector", "both"):
            parts.append(vector_retriever.retrieve(state["question"]))
        if route in ("graph", "both"):
            g = graph_retriever.retrieve(state["question"])
            if g:
                parts.append("Relacoes do grafo:\n" + g)
        
        context = "\n\n".join(p for p in parts if p)
        from llm_finetuning.guardrails import GUARDRAILS
        jb = GUARDRAILS.build("jailbreak_block")
        if context and not jb.apply(context, "input").allowed:
            context = ""
            
        return {"context": context}

    def generate(state: dict[str, Any]) -> dict[str, Any]:
        msgs = build_generator_messages(
            state["question"], state.get("context", ""), state.get("feedback", "")
        )
        return {"answer": llm.chat(msgs)}

    def critic(state: dict[str, Any]) -> dict[str, Any]:
        verdict = parse_critic(
            llm.chat(build_critic_messages(state["question"], state.get("context", ""), state["answer"]))
        )
        return {
            "approved": verdict["approved"],
            "feedback": verdict["feedback"],
            "reflections": state.get("reflections", 0) + 1,
            "corrected": state.get("corrected", False) or not verdict["approved"],
        }

    def after_critic(state: dict[str, Any]) -> str:
        if state["approved"] or state["reflections"] >= max_reflections:
            return END
        return "analyze"

    graph = StateGraph(RagState)
    graph.add_node("analyze", analyze)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("critic", critic)
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critic")
    graph.add_conditional_edges("critic", after_critic, {"analyze": "analyze", END: END})
    return graph.compile()
