"""Unit tests for the TrainingArguments plumbing of ContinualPretrainTrainer.

Exercises the pure kwargs builder so the FSDP / gradient-checkpointing / optimizer
options can be validated without importing torch or transformers.
"""

from __future__ import annotations

from llm_finetuning.training.pretrain import ContinualPretrainTrainer


def test_defaults_have_no_fsdp_or_checkpointing():
    trainer = ContinualPretrainTrainer()
    kwargs = trainer._training_arguments_kwargs()
    assert kwargs["gradient_checkpointing"] is False
    assert kwargs["fsdp"] == ""
    assert kwargs["optim"] == "adamw_torch"
    # No checkpointing kwargs when checkpointing is off.
    assert "gradient_checkpointing_kwargs" not in kwargs
    assert "fsdp_config" not in kwargs


def test_fsdp_and_checkpointing_are_passed_through():
    fsdp_config = {"transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"]}
    trainer = ContinualPretrainTrainer(
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        fsdp="full_shard auto_wrap",
        fsdp_config=fsdp_config,
    )
    kwargs = trainer._training_arguments_kwargs()
    assert kwargs["gradient_checkpointing"] is True
    assert kwargs["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert kwargs["optim"] == "adamw_torch_fused"
    assert kwargs["fsdp"] == "full_shard auto_wrap"
    assert kwargs["fsdp_config"] == fsdp_config


def test_defaults_do_not_checkpoint():
    kwargs = ContinualPretrainTrainer()._training_arguments_kwargs()
    assert kwargs["save_strategy"] == "no"
    assert "save_steps" not in kwargs
    assert "save_total_limit" not in kwargs


def test_step_checkpointing_is_passed_through():
    trainer = ContinualPretrainTrainer(
        save_strategy="steps", save_steps=500, save_total_limit=2,
    )
    kwargs = trainer._training_arguments_kwargs()
    assert kwargs["save_strategy"] == "steps"
    assert kwargs["save_steps"] == 500
    assert kwargs["save_total_limit"] == 2


def test_epoch_checkpointing_has_no_save_steps():
    kwargs = ContinualPretrainTrainer(save_strategy="epoch")._training_arguments_kwargs()
    assert kwargs["save_strategy"] == "epoch"
    assert "save_steps" not in kwargs  # steps cadence irrelevant for epoch saves
    assert kwargs["save_total_limit"] == 2


def test_core_hyperparameters_round_trip():
    trainer = ContinualPretrainTrainer(
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        bf16=True,
    )
    kwargs = trainer._training_arguments_kwargs()
    assert kwargs["learning_rate"] == 2e-5
    assert kwargs["per_device_train_batch_size"] == 4
    assert kwargs["gradient_accumulation_steps"] == 4
    assert kwargs["bf16"] is True
