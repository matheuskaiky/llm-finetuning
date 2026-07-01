"""Continued pre-training (Q1): causal-LM training on a domain text corpus.

The trainer concatenates the corpus into ``block_size`` token blocks and runs the
HuggingFace ``Trainer`` with a causal-LM collator. ``torch``, ``transformers`` and
``datasets`` are imported lazily inside :meth:`train`.
"""

from __future__ import annotations

from typing import Any

from ..core.interfaces import ModelBundle, Trainer, TrainResult
from ..core.registry import TRAINERS
from ..data.text_corpus import chunk_token_ids


@TRAINERS.register("continual_pretrain")
class ContinualPretrainTrainer(Trainer):
    """Continues pre-training a causal LM on a list of text documents.

    Args:
        output_dir: Where the checkpoint and logs are written.
        block_size: Token block length used to pack the corpus.
        num_train_epochs: Number of passes over the corpus.
        learning_rate: Optimizer learning rate (keep low to avoid forgetting).
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.
        weight_decay: AdamW weight decay.
        warmup_ratio: Fraction of steps used for LR warmup.
        logging_steps: Logging interval in steps.
        fp16: Enable fp16 mixed precision.
        bf16: Enable bf16 mixed precision.
        gradient_checkpointing: Recompute activations in the backward pass to
            trade compute for memory (mathematically identical result).
        optim: Optimizer name forwarded to TrainingArguments (e.g.
            ``"adamw_torch"``). Kept full-parameter; not a PEFT switch.
        fsdp: TrainingArguments FSDP mode string (e.g.
            ``"full_shard auto_wrap"``); empty disables sharding (single device).
        fsdp_config: FSDP options dict (e.g. layer class to wrap). Forwarded only
            when ``fsdp`` is set.
    """

    def __init__(
        self,
        output_dir: str = "outputs/pretrain",
        block_size: int = 512,
        num_train_epochs: float = 1.0,
        learning_rate: float = 5e-5,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.03,
        logging_steps: int = 10,
        fp16: bool = False,
        bf16: bool = False,
        gradient_checkpointing: bool = False,
        optim: str = "adamw_torch",
        fsdp: str = "",
        fsdp_config: dict[str, Any] | None = None,
        max_grad_norm: float = 1.0,
        save_strategy: str = "no",
        save_steps: int = 500,
        save_total_limit: int = 2,
    ) -> None:
        self.output_dir = output_dir
        self.max_grad_norm = max_grad_norm
        self.block_size = block_size
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.logging_steps = logging_steps
        self.fp16 = fp16
        self.bf16 = bf16
        self.gradient_checkpointing = gradient_checkpointing
        self.optim = optim
        self.fsdp = fsdp
        self.fsdp_config = fsdp_config
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit

    def _training_arguments_kwargs(self) -> dict[str, Any]:
        """Build the TrainingArguments kwargs (pure: no transformers import).

        Sharding (``fsdp``) and ``gradient_checkpointing`` only rearrange memory or
        recompute activations; they keep the run full-parameter and leave the
        optimization result unchanged. ``gradient_checkpointing_kwargs`` and
        ``fsdp_config`` are added only when their feature is enabled.
        """
        kwargs: dict[str, Any] = {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "logging_steps": self.logging_steps,
            "save_strategy": self.save_strategy,
            "report_to": [],
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optim": self.optim,
            "fsdp": self.fsdp,
            "max_grad_norm": self.max_grad_norm,
        }
        if self.gradient_checkpointing:
            kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        if self.fsdp and self.fsdp_config:
            kwargs["fsdp_config"] = self.fsdp_config
        # Periodic checkpointing protects long full-corpus runs from a time-limit
        # kill; "steps" also needs a save_steps cadence and a total limit.
        if self.save_strategy == "steps":
            kwargs["save_steps"] = self.save_steps
        if self.save_strategy != "no":
            kwargs["save_total_limit"] = self.save_total_limit
        return kwargs

    def _concatenate_token_ids(self, texts: list[str], tokenizer: Any) -> list[int]:
        """Tokenize documents and concatenate, inserting EOS between them."""
        eos_id = tokenizer.eos_token_id
        ids: list[int] = []
        for text in texts:
            if not text.strip():
                continue
            ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
            if eos_id is not None:
                ids.append(eos_id)
        return ids

    def _build_block_dataset(self, texts: list[str], tokenizer: Any) -> Any:
        from datasets import Dataset

        token_ids = self._concatenate_token_ids(texts, tokenizer)
        blocks = chunk_token_ids(token_ids, self.block_size, drop_remainder=True)
        if not blocks:
            raise ValueError(
                "corpus too small to form a single block of "
                f"{self.block_size} tokens"
            )
        return Dataset.from_dict({"input_ids": blocks})

    def train(self, model: ModelBundle, dataset: list[str], config: Any = None) -> TrainResult:
        from transformers import (
            DataCollatorForLanguageModeling,
            TrainingArguments,
        )
        from transformers import Trainer as HfTrainer

        net, tokenizer = model
        block_dataset = self._build_block_dataset(dataset, tokenizer)
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        if self.gradient_checkpointing and hasattr(net, "config"):
            # KV cache is incompatible with activation checkpointing during training.
            net.config.use_cache = False

        args = TrainingArguments(**self._training_arguments_kwargs())
        hf_trainer = HfTrainer(
            model=net,
            args=args,
            train_dataset=block_dataset,
            data_collator=collator,
        )
        outcome = hf_trainer.train()
        hf_trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        metrics = {
            "train_loss": float(outcome.training_loss),
            "num_blocks": len(block_dataset),
        }
        return TrainResult(model=net, metrics=metrics, output_dir=self.output_dir)
