#!/usr/bin/env python3
"""Download the base model and the training dataset from the Hugging Face Hub.

Reads BASE_MODEL_ID and DATASET_ID from the environment (or a local .env), with
CLI overrides. Files land under models/ and data/raw/, both git-ignored.

Usage:
    python scripts/download_assets.py --all
    python scripts/download_assets.py --model
    python scripts/download_assets.py --dataset --dataset-id gutoportelaa/dom-pi-corpus-2025
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_DATASET_ID = "gutoportelaa/dom-pi-corpus-2025"


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Populate os.environ from a .env file for keys not already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def download_model(model_id: str, out_root: Path = Path("models")) -> Path:
    from huggingface_hub import snapshot_download

    dest = out_root / model_id.split("/")[-1]
    snapshot_download(repo_id=model_id, repo_type="model", local_dir=dest)
    print(f"model ready: {dest}")
    return dest


def download_dataset(dataset_id: str, out_root: Path = Path("data/raw")) -> Path:
    from huggingface_hub import snapshot_download

    dest = out_root / dataset_id.split("/")[-1]
    snapshot_download(repo_id=dataset_id, repo_type="dataset", local_dir=dest)
    print(f"dataset ready: {dest}")
    return dest


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="download model and dataset")
    parser.add_argument("--model", action="store_true", help="download the base model")
    parser.add_argument("--dataset", action="store_true", help="download the dataset")
    parser.add_argument(
        "--model-id", default=os.environ.get("BASE_MODEL_ID", DEFAULT_MODEL_ID)
    )
    parser.add_argument(
        "--dataset-id", default=os.environ.get("DATASET_ID", DEFAULT_DATASET_ID)
    )
    args = parser.parse_args()

    if not (args.all or args.model or args.dataset):
        parser.error("choose at least one of --all, --model, --dataset")

    if args.all or args.model:
        download_model(args.model_id)
    if args.all or args.dataset:
        download_dataset(args.dataset_id)


if __name__ == "__main__":
    main()
