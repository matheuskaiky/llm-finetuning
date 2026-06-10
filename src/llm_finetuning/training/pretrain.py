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
    ) -> None:
        self.output_dir = output_dir
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

        args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            logging_steps=self.logging_steps,
            save_strategy="no",
            report_to=[],
            fp16=self.fp16,
            bf16=self.bf16,
        )
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
