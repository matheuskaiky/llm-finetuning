"""Unit tests for SupervisedFineTuneTrainer pure logic (no ML stack)."""

from __future__ import annotations

from llm_finetuning.training.sft import (
    DEFAULT_LORA_TARGETS,
    SupervisedFineTuneTrainer,
    build_lora_kwargs,
)


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


def test_build_lora_kwargs_defaults_and_overrides():
    kw = build_lora_kwargs({})
    assert kw["r"] == 16 and kw["lora_alpha"] == 32 and kw["task_type"] == "CAUSAL_LM"
    assert kw["target_modules"] == DEFAULT_LORA_TARGETS and kw["bias"] == "none"
    kw2 = build_lora_kwargs({"r": 8, "alpha": 16, "dropout": 0.1,
                             "target_modules": ["q_proj"]})
    assert kw2["r"] == 8 and kw2["lora_alpha"] == 16 and kw2["lora_dropout"] == 0.1
    assert kw2["target_modules"] == ["q_proj"]


def test_peft_trainer_registered_and_optional():
    t_full = SupervisedFineTuneTrainer()
    t_lora = SupervisedFineTuneTrainer(peft={"r": 8})
    assert t_full.peft is None and t_lora.peft == {"r": 8}


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
