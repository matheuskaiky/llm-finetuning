#!/usr/bin/env python3
"""Run the baseline language-model evaluation from a config file.

Usage:
    python scripts/evaluate.py --config configs/eval_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_finetuning.core import instantiate, load_config, set_global_seed
from llm_finetuning.core.registry import MODEL_PROVIDERS
from llm_finetuning.evaluation.evaluator import LanguageModelEvaluator, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        help="override model.params.model_name (evaluate a different checkpoint)",
    )
    parser.add_argument(
        "--output-path", help="override evaluation.output_path for this run"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(config.seed)

    if config.model is None or config.evaluation is None:
        raise SystemExit("config must define both 'model' and 'evaluation' sections")

    # CLI overrides keep one config reusable across models/checkpoints.
    if args.model_name:
        config.model.params["model_name"] = args.model_name
    if args.output_path:
        config.evaluation.output_path = args.output_path

    provider = instantiate(MODEL_PROVIDERS, config.model)
    model_bundle = provider.load()

    evaluator = LanguageModelEvaluator.from_config(config.evaluation)
    results = evaluator.evaluate(model_bundle, config.evaluation.benchmark)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if config.evaluation.output_path:
        path = save_results(results, config.evaluation.output_path)
        print(f"\nsaved results -> {path}")


if __name__ == "__main__":
    main()
