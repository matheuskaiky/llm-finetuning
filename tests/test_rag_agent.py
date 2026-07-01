"""Unit tests for the pure agent/judge logic (no LLM, no langgraph import)."""

from __future__ import annotations

import pytest

from llm_finetuning.rag.agent import build_agent, decide_route, parse_critic, standard_rag
from llm_finetuning.rag.judge import parse_score
from llm_finetuning.rag.retrievers import candidate_entity_terms


def test_decide_route_semantic_vs_relational():
    assert decide_route("O que diz o decreto sobre ferias?") == "vector"
    assert decide_route("Quais empresas venceram licitacao da prefeitura?") == "both"


def test_decide_route_force_both_on_reflection():
    assert decide_route("pergunta simples", force_both=True) == "both"


def test_parse_critic_approved_and_rejected():
    assert parse_critic('{"approved": true, "feedback": ""}')["approved"] is True
    rej = parse_critic('{"approved": false, "feedback": "faltou citar o valor"}')
    assert rej["approved"] is False
    assert "valor" in rej["feedback"]


def test_parse_critic_unparseable_defaults_approved():
    # Avoids trapping the loop when the critic returns noise.
    assert parse_critic("blah blah") ["approved"] is True


def test_parse_score_bounds():
    assert parse_score("5") == 5
    assert parse_score("a nota e 3 de 5") == 3
    assert parse_score("9") == 5
    assert parse_score("nenhum numero") == 0
    assert parse_score("-2") == 0


def test_candidate_entity_terms_drops_stopwords():
    terms = candidate_entity_terms("Quais empresas tem contrato com a Prefeitura de Teresina?")
    lowered = [t.casefold() for t in terms]
    assert "quais" not in lowered and "com" not in lowered
    assert "empresas" in lowered and "teresina" in lowered


class _StubLLM:
    """Critic rejects once then approves; generator returns a fixed answer."""

    def __init__(self) -> None:
        self.critic_calls = 0

    def chat(self, messages, max_new_tokens=None):
        if "avaliar" in messages[-1]["content"]:
            self.critic_calls += 1
            approved = "false" if self.critic_calls == 1 else "true"
            return f'{{"approved": {approved}, "feedback": "cite o valor"}}'
        return "Empresa X venceu a licitacao."


class _StubRetriever:
    def __init__(self, text: str) -> None:
        self._text = text

    def retrieve(self, question: str) -> str:
        return self._text


def test_standard_rag_is_single_shot_vector_only():
    llm = _StubLLM()
    graph = _StubRetriever("GRAPH-CONTEXT-SHOULD-NOT-APPEAR")
    out = standard_rag(llm, _StubRetriever("[d1] contexto vetorial"), "pergunta?")
    assert out  # produced an answer
    # standard_rag never touches the graph retriever (single vector shot).
    assert graph.retrieve("x") == "GRAPH-CONTEXT-SHOULD-NOT-APPEAR"  # still callable, just unused


def test_runner_registry_builds_each_mode():
    pytest.importorskip("langgraph")
    from llm_finetuning.rag.pipelines import RUNNERS, build_runner

    assert set(RUNNERS) == {"standard", "agentic_vector", "agentic_graph"}
    llm, vec, gra = _StubLLM(), _StubRetriever("[d1] ctx"), _StubRetriever("A -[r]-> B")
    for mode in RUNNERS:
        runner = build_runner(mode, llm, vec, gra, max_reflections=1)
        res = runner.answer("Quais empresas venceram licitacao?")
        assert res.answer
        assert isinstance(res.corrected, bool)


def test_agentic_vector_runner_never_uses_graph():
    pytest.importorskip("langgraph")
    from llm_finetuning.rag.pipelines import build_runner

    class _GraphProbe:
        used = False

        def retrieve(self, q):
            _GraphProbe.used = True
            return "graph!"

    runner = build_runner("agentic_vector", _StubLLM(), _StubRetriever("[d1] ctx"), _GraphProbe(), 1)
    runner.answer("Quais empresas venceram licitacao da prefeitura?")  # relational
    assert _GraphProbe.used is False  # vector-only mode must skip the graph


def test_agent_self_reflection_loop_runs():
    pytest.importorskip("langgraph")
    agent = build_agent(
        _StubLLM(),
        _StubRetriever("[d1] contexto"),
        _StubRetriever("Empresa X -[venceu]-> Licitacao 01"),
        max_reflections=2,
    )
    out = agent.invoke({"question": "Quais empresas venceram licitacao da prefeitura?"})
    assert out["route"] == "both"  # relational question
    assert out["corrected"] is True  # critic forced a retry
    assert out["reflections"] == 2
    assert "Empresa X" in out["answer"]
