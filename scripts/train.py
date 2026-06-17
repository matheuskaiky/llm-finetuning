#!/usr/bin/env python3
"""Run a training experiment from a config file.

Builds the model, dataset and trainer from the config registries, optionally
evaluates the model before and after training, and saves the results.

Usage:
    python scripts/train.py --config configs/pretrain_diarios.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from llm_finetuning.core import instantiate, load_config, set_global_seed
from llm_finetuning.core.config import EvaluationConfig
from llm_finetuning.core.registry import DATASET_LOADERS, MODEL_PROVIDERS, TRAINERS
from llm_finetuning.evaluation.evaluator import LanguageModelEvaluator, save_results


def _evaluate(model_bundle: object, eval_config: EvaluationConfig) -> dict[str, float]:
    evaluator = LanguageModelEvaluator.from_config(eval_config)
    return evaluator.evaluate(model_bundle, eval_config.benchmark)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model-name", help="override model.params.model_name (e.g. a Q1 checkpoint)"
    )
    parser.add_argument(
        "--output-dir", help="override trainer.params.output_dir for this run"
    )
    parser.add_argument(
        "--data-path", help="override dataset.params.path (e.g. a per-teacher distill set)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(config.seed)

    if config.model is None or config.dataset is None or config.trainer is None:
        raise SystemExit("config must define 'model', 'dataset' and 'trainer'")

    # CLI overrides keep one config reusable across starting checkpoints/outputs.
    if args.model_name:
        config.model.params["model_name"] = args.model_name
    if args.output_dir:
        config.trainer.params["output_dir"] = args.output_dir
    if args.data_path:
        config.dataset.params["path"] = args.data_path

    # Under a distributed launch (torchrun/accelerate, WORLD_SIZE>1) the model is
    # FSDP-sharded inside the Trainer, so in-process before/after evaluation is not
    # run here: evaluate separately with scripts/evaluate.py on the base model and
    # on the saved checkpoint. Only rank 0 prints/saves.
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    is_main = int(os.environ.get("RANK", "0")) == 0

    provider = instantiate(MODEL_PROVIDERS, config.model)
    model_bundle = provider.load()
    loader = instantiate(DATASET_LOADERS, config.dataset)
    documents = loader.load()
    trainer = instantiate(TRAINERS, config.trainer)

    run_in_process_eval = config.evaluation is not None and not is_distributed

    before = None
    if run_in_process_eval:
        before = _evaluate(model_bundle, config.evaluation)
        print("eval before:", json.dumps(before, ensure_ascii=False))

    result = trainer.train(model_bundle, documents, config)
    if is_main:
        print(f"training done: output_dir={result.output_dir} metrics={result.metrics}")

    if run_in_process_eval:
        after = _evaluate(model_bundle, config.evaluation)
        print("eval after:", json.dumps(after, ensure_ascii=False))
        if config.evaluation.output_path:
            path = save_results(
                {"before": before, "after": after, "train": result.metrics},
                config.evaluation.output_path,
            )
            print(f"saved results: {path}")


if __name__ == "__main__":
    main()
