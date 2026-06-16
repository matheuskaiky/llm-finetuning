"""Unit tests for SupervisedFineTuneTrainer pure logic (no ML stack)."""

from __future__ import annotations

from llm_finetuning.training.sft import SupervisedFineTuneTrainer


class _FakeTok:
    """Minimal tokenizer: maps each whitespace token to a stable id; eos=99."""

    eos_token_id = 99

    def __call__(self, text, add_special_tokens=True):
        ids = [(abs(hash(w)) % 90) + 1 for w in text.split()]
        return {"input_ids": ids}


def test_training_arguments_kwargs_registered_and_built():
    from llm_finetuning.core.registry import TRAINERS

    assert "sft" in TRAINERS  # registered by import side effect
    t = SupervisedFineTuneTrainer(output_dir="x", bf16=True, gradient_checkpointing=True)
    kw = t._training_arguments_kwargs()
    assert kw["output_dir"] == "x" and kw["bf16"] is True
    assert kw["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert kw["save_strategy"] == "no"


def test_encode_masks_prompt_and_keeps_response():
    t = SupervisedFineTuneTrainer(max_length=64)
    enc = t._encode({"instruction": "defina pilha", "input": "", "output": "uma estrutura"}, _FakeTok())
    assert enc["input_ids"][-1] == 99  # eos appended
    assert len(enc["input_ids"]) == len(enc["labels"]) == len(enc["attention_mask"])
    # at least one masked prompt position and one trainable response position
    assert -100 in enc["labels"]
    assert any(t_ != -100 for t_ in enc["labels"])
    # the trailing labels (response + eos) match the trailing input_ids
    assert enc["labels"][-1] == enc["input_ids"][-1] == 99
