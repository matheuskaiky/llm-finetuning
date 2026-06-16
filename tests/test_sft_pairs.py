"""Unit tests for the Q2 SFT data helpers (pure, no ML stack)."""

from __future__ import annotations

from llm_finetuning.data.sft_pairs import (
    SftPairsLoader,
    build_input_and_labels,
    build_prompt,
    clean_source_records,
    dedup_pairs,
    is_clean_record,
    parse_qa_pairs,
    subtract_pairs,
    weird_letter_ratio,
)

_PT = "Notas de aula sobre algoritmos de ordenacao e complexidade. Exemplo prático."
_GARBAGE = "ŗ˦ƉɄȳ óŗȭſǶȳƨțțǶ ɄȭɱʯʚŗƍźɄ ƨ Èϙã†È�English"


def test_weird_letter_ratio_low_for_portuguese_and_code():
    assert weird_letter_ratio(_PT) < 0.05
    assert weird_letter_ratio("for i in range(10): print(i)  # código") < 0.05


def test_weird_letter_ratio_high_for_glyph_garbage():
    assert weird_letter_ratio(_GARBAGE) > 0.5


def test_is_clean_record_filters_short_and_garbage():
    assert is_clean_record(_PT)
    assert not is_clean_record("curto")  # < min_chars
    assert not is_clean_record(_GARBAGE)


def test_clean_source_records_dedups_and_filters():
    recs = [
        {"text": _PT, "nome_professor": "A"},
        {"text": _PT, "nome_professor": "A"},  # exact dup
        {"text": _GARBAGE, "nome_professor": "B"},  # garbage
        {"text": "abc", "nome_professor": "C"},  # too short
        {"text": _PT + " Outro paragrafo distinto aqui.", "nome_professor": "A"},
    ]
    out = clean_source_records(recs)
    assert len(out) == 2  # the dup, garbage and short ones dropped
    assert all(r["text"].strip() for r in out)


def test_parse_qa_pairs_json_array():
    raw = (
        'aqui: [{"instruction": "O que e X?", "input": "", "output": "X e Y."}, '
        '{"instruction": "Defina Z", "output": "Z e W."}] fim'
    )
    pairs = parse_qa_pairs(raw)
    assert len(pairs) == 2
    assert pairs[0] == {"instruction": "O que e X?", "input": "", "output": "X e Y."}
    assert pairs[1]["input"] == ""  # missing input defaults to empty


def test_parse_qa_pairs_skips_malformed_and_empty():
    raw = '[{"instruction": "", "output": "sem pergunta"}, {"instruction": "ok", "output": ""}]'
    assert parse_qa_pairs(raw) == []
    assert parse_qa_pairs("texto sem json") == []


def test_dedup_pairs_by_instruction_and_input():
    pairs = [
        {"instruction": "O que e X?", "input": "", "output": "a"},
        {"instruction": "o que e x?  ", "input": "", "output": "b"},  # same, normalized
        {"instruction": "O que e X?", "input": "ctx", "output": "c"},  # different input
    ]
    out = dedup_pairs(pairs)
    assert len(out) == 2


def test_subtract_pairs_removes_training_questions():
    train = [{"instruction": "O que e X?", "input": "", "output": "a"}]
    cand = [
        {"instruction": "o que e x?", "input": "", "output": "b"},  # same as train
        {"instruction": "O que e Y?", "input": "", "output": "c"},  # new
    ]
    out = subtract_pairs(cand, train)
    assert len(out) == 1 and out[0]["instruction"] == "O que e Y?"


def test_build_prompt_with_and_without_input():
    p0 = build_prompt("Defina pilha", "")
    assert "### Instrucao:" in p0 and "Defina pilha" in p0 and "### Resposta:" in p0
    assert "### Entrada:" not in p0
    p1 = build_prompt("Resuma", "texto base")
    assert "### Entrada:" in p1 and "texto base" in p1


def test_build_input_and_labels_masks_prompt_and_appends_eos():
    ids, labels = build_input_and_labels([1, 2, 3], [4, 5], eos_id=9)
    assert ids == [1, 2, 3, 4, 5, 9]
    assert labels == [-100, -100, -100, 4, 5, 9]  # prompt masked, response + eos kept


def test_build_input_and_labels_truncates():
    ids, labels = build_input_and_labels([1, 2, 3], [4, 5], eos_id=9, max_length=4)
    assert ids == [1, 2, 3, 4]
    assert labels == [-100, -100, -100, 4]
    assert len(ids) == len(labels) == 4


def test_sft_pairs_loader_roundtrip(tmp_path):
    import json

    f = tmp_path / "pairs.jsonl"
    f.write_text(
        json.dumps({"instruction": "a", "input": "", "output": "b"}, ensure_ascii=False)
        + "\n\n"  # blank line ignored
        + json.dumps({"instruction": "c", "output": "d"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    rows = SftPairsLoader(f).load()
    assert len(rows) == 2
    assert rows[0]["instruction"] == "a" and rows[1]["output"] == "d"
