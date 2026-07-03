"""Unit tests for the Q2 source-anchored deep-question dataset helpers.

Pure, no ML stack. Cleaning/parsing/dedup are covered by test_sft_pairs.py and
only imported here where reused.
"""

from __future__ import annotations

import pytest

from llm_finetuning.data.anchored_qa_pairs import (
    GenerationAborted,
    archive_existing_outputs,
    build_user_message,
    generate_pairs_for_records,
    is_deep_enough,
    to_sft_record,
    to_source_record,
)


class FakeLLM:
    """Duck-typed stand-in for LocalChatLLM: scripted replies or forced errors."""

    def __init__(self, replies=None, raise_on=()):
        self.replies = list(replies or [])
        self.raise_on = set(raise_on)
        self.calls = 0

    def chat(self, messages, max_new_tokens):
        call_index = self.calls
        self.calls += 1
        if call_index in self.raise_on:
            raise RuntimeError(f"boom at call {call_index}")
        return self.replies[call_index] if call_index < len(self.replies) else "[]"


def test_is_deep_enough_rejects_shallow_openers():
    shallow = {"instruction": "O que é uma pilha?", "output": "Uma pilha e uma estrutura LIFO."}
    assert not is_deep_enough(shallow)


def test_is_deep_enough_rejects_shallow_openers_case_and_accent_insensitive():
    for instruction in ("o que e recursao?", "Defina recursao", "PARA QUE SERVE isso?"):
        assert not is_deep_enough({"instruction": instruction, "output": "resposta longa o bastante"})


def test_is_deep_enough_rejects_short_or_echoed_output():
    assert not is_deep_enough({"instruction": "Por que X falha?", "output": "curta"})
    assert not is_deep_enough({"instruction": "mesma coisa", "output": "mesma coisa"})


def test_is_deep_enough_accepts_grounded_deep_pair():
    pair = {
        "instruction": "Por que o algoritmo do trecho prefere quicksort a bubble sort nesse caso?",
        "output": "Porque o trecho descreve um cenario com dados quase ordenados, "
                  "onde o quicksort evita o pior caso O(n^2) citado no material.",
    }
    assert is_deep_enough(pair)


def test_build_user_message_format():
    msg = build_user_message("Fulano", "Trecho de exemplo.")
    assert msg == "Professor: Fulano\nTrecho:\nTrecho de exemplo.\n\nJSON:"


_GOOD = '[{"instruction": "Por que o trecho prefere X a Y?", "input": "", ' \
        '"output": "Porque o trecho descreve um cenario em que X evita o problema citado."}]'
_GOOD2 = '[{"instruction": "Qual o proximo passo do procedimento no trecho?", "input": "", ' \
         '"output": "O trecho descreve que o proximo passo e validar a entrada antes do calculo."}]'
_SHALLOW = '[{"instruction": "O que e X?", "output": "X e uma estrutura de dados."}]'
_RECORDS = [{"text": f"trecho {i}", "nome_professor": f"prof{i}"} for i in range(10)]


def test_generate_pairs_skips_malformed_llm_output_and_keeps_going():
    llm = FakeLLM(replies=["isso nao e json nenhum", _GOOD, _GOOD2])
    pairs, sources, stats = generate_pairs_for_records(llm, _RECORDS, target=2, max_chars=100,
                                                        max_new_tokens=64)
    assert len(pairs) == 2
    assert stats["malformed"] == 1
    assert stats["calls"] == 3  # 1 malformed + 2 good, stopped once target hit


def test_generate_pairs_rejects_shallow_pairs_via_is_deep_enough():
    llm = FakeLLM(replies=[_SHALLOW, _GOOD])
    pairs, sources, stats = generate_pairs_for_records(llm, _RECORDS, target=1, max_chars=100,
                                                        max_new_tokens=64)
    assert len(pairs) == 1
    assert stats["shallow"] == 1


def test_generate_pairs_survives_isolated_llm_chat_exception():
    # call 0 raises (content ignored), calls 1 and 2 return the two good pairs
    llm = FakeLLM(replies=["ignored", _GOOD, _GOOD2], raise_on={0})
    pairs, sources, stats = generate_pairs_for_records(llm, _RECORDS, target=2, max_chars=100,
                                                        max_new_tokens=64)
    assert len(pairs) == 2
    assert stats["errors"] == 1


def test_generate_pairs_source_records_stay_aligned_with_sft_pairs():
    llm = FakeLLM(replies=[_GOOD])
    pairs, sources, stats = generate_pairs_for_records(llm, _RECORDS, target=1, max_chars=100,
                                                        max_new_tokens=64)
    assert sources[0]["instruction"] == pairs[0]["instruction"]
    assert sources[0]["professor"] == "prof0"
    assert sources[0]["doc_index"] == 0


def test_generate_pairs_aborts_after_consecutive_llm_chat_failures():
    llm = FakeLLM(replies=[], raise_on={0, 1, 2})
    with pytest.raises(GenerationAborted):
        generate_pairs_for_records(llm, _RECORDS, target=5, max_chars=100, max_new_tokens=64,
                                    abort_after_consecutive_errors=3)


def test_generate_pairs_does_not_abort_when_errors_are_not_consecutive():
    # calls: 0 raises, 1 good, 2 raises, 3 good - errors never repeat back-to-back
    llm = FakeLLM(replies=["ignored", _GOOD, "ignored", _GOOD2], raise_on={0, 2})
    pairs, sources, stats = generate_pairs_for_records(llm, _RECORDS, target=2, max_chars=100,
                                                        max_new_tokens=64,
                                                        abort_after_consecutive_errors=2)
    assert len(pairs) == 2
    assert stats["errors"] == 2


def test_to_sft_record_drops_source_fields():
    pair = {"instruction": "Por que X?", "input": "", "output": "Porque Y."}
    rec = to_sft_record(pair)
    assert rec == {"instruction": "Por que X?", "input": "", "output": "Porque Y."}
    assert "professor" not in rec and "doc_index" not in rec


def test_to_sft_record_defaults_missing_input():
    rec = to_sft_record({"instruction": "Q", "output": "A"})
    assert rec["input"] == ""


def test_to_source_record_includes_traceability_fields():
    pair = {"instruction": "Por que X?", "input": "", "output": "Porque Y."}
    rec = to_source_record(pair, professor="Fulano", doc_index=42, excerpt="trecho...")
    assert rec == {
        "instruction": "Por que X?",
        "input": "",
        "output": "Porque Y.",
        "professor": "Fulano",
        "doc_index": 42,
        "source_excerpt": "trecho...",
    }


def test_archive_existing_outputs_moves_files(tmp_path):
    out_dir = tmp_path / "sft"
    out_dir.mkdir()
    old_dir = out_dir / "old"
    f = out_dir / "docentes_sft_train.jsonl"
    f.write_text("old content", encoding="utf-8")

    archive_existing_outputs([f], old_dir)

    assert not f.exists()
    assert (old_dir / "docentes_sft_train.jsonl").read_text(encoding="utf-8") == "old content"


def test_archive_existing_outputs_skips_missing_files(tmp_path):
    out_dir = tmp_path / "sft"
    out_dir.mkdir()
    old_dir = out_dir / "old"
    missing = out_dir / "does_not_exist.jsonl"

    archive_existing_outputs([missing], old_dir)  # must not raise

    assert not old_dir.exists()


def test_archive_existing_outputs_overwrites_previous_archive(tmp_path):
    out_dir = tmp_path / "sft"
    out_dir.mkdir()
    old_dir = out_dir / "old"
    old_dir.mkdir()
    (old_dir / "docentes_sft_train.jsonl").write_text("stale archive", encoding="utf-8")
    f = out_dir / "docentes_sft_train.jsonl"
    f.write_text("new run content", encoding="utf-8")

    archive_existing_outputs([f], old_dir)

    assert (old_dir / "docentes_sft_train.jsonl").read_text(encoding="utf-8") == "new run content"
