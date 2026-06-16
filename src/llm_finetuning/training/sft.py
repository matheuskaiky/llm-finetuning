"""Supervised fine-tuning (Q2): instruction tuning on generated Q&A pairs.

The trainer renders each ``{instruction, input?, output}`` pair with the SFT
template, tokenizes prompt and response separately, and masks the loss to the
response tokens only (completion-only SFT). ``torch``/``transformers``/``datasets``
are imported lazily inside :meth:`train`. Full-parameter by default; Q3 reuses this
trainer with LoRA/QLoRA via configuration.
"""

from __future__ import annotations

from typing import Any

from ..core.interfaces import ModelBundle, Trainer, TrainResult
from ..core.registry import TRAINERS
from ..data.sft_pairs import build_input_and_labels, build_prompt


@TRAINERS.register("sft")
class SupervisedFineTuneTrainer(Trainer):
    """Supervised fine-tuning of a causal LM on instruction pairs.

    Args mirror :class:`ContinualPretrainTrainer`, with ``max_length`` (token cap
    per example) instead of ``block_size``.
    """

    def __init__(
        self,
        output_dir: str = "outputs/sft",
        max_length: int = 1024,
        num_train_epochs: float = 3.0,
        learning_rate: float = 2e-5,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        weight_decay: float = 0.0,
        warmup_ratio: float = 0.03,
        logging_steps: int = 10,
        fp16: bool = False,
        bf16: bool = False,
        gradient_checkpointing: bool = False,
        optim: str = "adamw_torch",
        fsdp: str = "",
        fsdp_config: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.max_length = max_length
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

    def _training_arguments_kwargs(self) -> dict[str, Any]:
        """Build the TrainingArguments kwargs (pure: no transformers import)."""
        kwargs: dict[str, Any] = {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "logging_steps": self.logging_steps,
            "save_strategy": "no",
            "report_to": [],
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optim": self.optim,
            "fsdp": self.fsdp,
        }
        if self.gradient_checkpointing:
            kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        if self.fsdp and self.fsdp_config:
            kwargs["fsdp_config"] = self.fsdp_config
        return kwargs

    def _encode(self, pair: dict, tokenizer: Any) -> dict:
        prompt = build_prompt(pair.get("instruction", ""), pair.get("input", ""))
        output = (pair.get("output", "") or "").strip()
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        output_ids = tokenizer(output, add_special_tokens=False)["input_ids"]
        input_ids, labels = build_input_and_labels(
            prompt_ids, output_ids, tokenizer.eos_token_id, self.max_length
        )
        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": [1] * len(input_ids)}

    def _build_dataset(self, pairs: list[dict], tokenizer: Any) -> Any:
        from datasets import Dataset

        rows = [self._encode(p, tokenizer) for p in pairs]
        rows = [r for r in rows if any(t != -100 for t in r["labels"])]
        if not rows:
            raise ValueError("no trainable SFT examples after encoding")
        return Dataset.from_list(rows)

    def train(self, model: ModelBundle, dataset: list[dict], config: Any = None) -> TrainResult:
        from transformers import DataCollatorForSeq2Seq, TrainingArguments
        from transformers import Trainer as HfTrainer

        net, tokenizer = model
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_dataset = self._build_dataset(dataset, tokenizer)
        collator = DataCollatorForSeq2Seq(
            tokenizer, label_pad_token_id=-100, padding=True
        )

        if self.gradient_checkpointing and hasattr(net, "config"):
            net.config.use_cache = False

        args = TrainingArguments(**self._training_arguments_kwargs())
        hf_trainer = HfTrainer(
            model=net,
            args=args,
            train_dataset=train_dataset,
            data_collator=collator,
        )
        outcome = hf_trainer.train()
        hf_trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        metrics = {
            "train_loss": float(outcome.training_loss),
            "num_examples": len(train_dataset),
        }
        return TrainResult(model=net, metrics=metrics, output_dir=self.output_dir)
