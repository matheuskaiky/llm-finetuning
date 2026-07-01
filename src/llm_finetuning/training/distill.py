"""Logit-based knowledge distillation (Q4): KL(teacher||student) + CE.

Soft-label distillation: the student matches the teacher's next-token distribution
(temperature-scaled KL) on the response tokens, plus the usual hard-label CE.
Teacher and student must share the tokenizer/vocabulary (same family, e.g.
Qwen3-1.7B -> Qwen3-0.6B). Reuses the SFT dataset building (prompt/response masking).
``torch``/``transformers`` are imported lazily inside :meth:`train`.
"""

from __future__ import annotations

from typing import Any

from ..core.interfaces import ModelBundle, TrainResult
from ..core.registry import TRAINERS
from .sft import SupervisedFineTuneTrainer


def kd_loss(student_logits, teacher_logits, labels, temperature: float):
    """Temperature-scaled KL(teacher||student) over response tokens (pure torch).

    ``*_logits`` are ``[B, T, V]``; ``labels`` is ``[B, T]`` with ``-100`` on the
    prompt. Applies the causal shift, masks to label!=-100, averages over those
    positions and scales by ``T^2`` (standard KD).
    """
    import torch.nn.functional as F

    s = student_logits[:, :-1, :]
    t = teacher_logits[:, :-1, :]
    lbl = labels[:, 1:]
    mask = lbl != -100
    T = temperature
    s_logp = F.log_softmax(s / T, dim=-1)
    t_p = F.softmax(t / T, dim=-1)
    # F.kl_div(log_q, p) = sum p*(log p - log q) = KL(p||q), per element; sum vocab.
    kl = F.kl_div(s_logp, t_p, reduction="none").sum(-1)  # [B, T-1]
    denom = mask.sum().clamp(min=1)
    return (kl * mask).sum() / denom * (T * T)


@TRAINERS.register("logit_kd")
class LogitKDTrainer(SupervisedFineTuneTrainer):
    """Distill a teacher into a student via logit KD + CE on the SFT pairs.

    Args extend the SFT trainer with ``teacher_model`` (path), ``kd_alpha`` (weight
    of the KD term vs CE) and ``kd_temperature``.
    """

    def __init__(
        self,
        teacher_model: str = "models/Qwen3-1.7B-Base",
        kd_alpha: float = 0.5,
        kd_temperature: float = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature

    def train(self, model: ModelBundle, dataset: list[dict], config: Any = None) -> TrainResult:
        import torch
        from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments
        from transformers import Trainer as HfTrainer

        net, tokenizer = model
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_dataset = self._build_dataset(dataset, tokenizer)
        collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, padding=True)

        teacher = AutoModelForCausalLM.from_pretrained(
            self.teacher_model, torch_dtype="auto"
        )
        teacher.eval()
        if torch.cuda.is_available():
            teacher = teacher.to(net.device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        if net.config.vocab_size != teacher.config.vocab_size:
            raise ValueError(
                "logit-KD requires the same vocab: student "
                f"{net.config.vocab_size} vs teacher {teacher.config.vocab_size}"
            )

        if self.gradient_checkpointing and hasattr(net, "config"):
            net.config.use_cache = False

        alpha, temp = self.kd_alpha, self.kd_temperature

        class _KDTrainer(HfTrainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kw):
                out = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask"),
                            labels=inputs["labels"])
                ce = out.loss
                with torch.no_grad():
                    t_logits = teacher(input_ids=inputs["input_ids"],
                                       attention_mask=inputs.get("attention_mask")).logits
                kd = kd_loss(out.logits, t_logits, inputs["labels"], temp)
                loss = alpha * kd + (1.0 - alpha) * ce
                return (loss, out) if return_outputs else loss

        args = TrainingArguments(**self._training_arguments_kwargs())
        hf_trainer = _KDTrainer(model=net, args=args, train_dataset=train_dataset,
                                data_collator=collator)
        outcome = hf_trainer.train()
        hf_trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        metrics = {
            "train_loss": float(outcome.training_loss),
            "num_examples": len(train_dataset),
            "method": "logit_kd",
            "teacher": self.teacher_model,
        }
        return TrainResult(model=net, metrics=metrics, output_dir=self.output_dir)
