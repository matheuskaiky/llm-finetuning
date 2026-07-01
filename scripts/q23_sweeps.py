#!/usr/bin/env python3
"""Differentiator experiments (node02): Q3 LoRA-rank sweep and Q2 data-size curve.

Trains several small variants of Qwen3-0.6B on the docentes SFT data so the report
can plot quality vs rank (Q3) and quality vs number of pairs (Q2). Evaluation is done
separately with scripts/eval_sft.py on the saved checkpoints.

    python scripts/q23_sweeps.py rank   # LoRA r in {4,8,16,32,64}
    python scripts/q23_sweeps.py data   # full SFT on n in {250,500,1000} pairs
"""
from __future__ import annotations

import sys

from llm_finetuning.data.sft_pairs import SftPairsLoader
from llm_finetuning.models.providers import LocalModelProvider
from llm_finetuning.training.sft import SupervisedFineTuneTrainer

DATA = "data/processed/sft/docentes_sft_train.jsonl"


def train(out: str, pairs: list[dict], peft=None) -> None:
    mb = LocalModelProvider(model_name="models/Qwen3-0.6B-Base", dtype="bfloat16").load()
    t = SupervisedFineTuneTrainer(
        output_dir=out, peft=peft, max_length=1024, num_train_epochs=3,
        learning_rate=2e-4 if peft else 2e-5, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, bf16=True, gradient_checkpointing=True)
    t.train(mb, pairs)
    print(f"== trained {out} ==", flush=True)


def main() -> None:
    pairs = SftPairsLoader(DATA).load()
    mode = sys.argv[1] if len(sys.argv) > 1 else "rank"
    if mode == "rank":
        for r in [4, 8, 16, 32, 64]:
            train(f"outputs/q3_rank_r{r}", pairs, peft={"r": r, "alpha": 2 * r, "dropout": 0.05})
    elif mode == "data":
        for n in [250, 500, 1000]:
            train(f"outputs/q2_data_n{n}", pairs[:n])
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
