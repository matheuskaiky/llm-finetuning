#!/usr/bin/env python3
"""Convert a directory of gazette PDFs into normalized ``.txt`` files.

Usage:
    python scripts/pdf_to_text.py --in data/raw --out data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_finetuning.data import PdfToTextLoader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", dest="output_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--boilerplate-threshold", type=float, default=0.6)
    args = parser.parse_args()

    loader = PdfToTextLoader(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        boilerplate_threshold=args.boilerplate_threshold,
    )
    outputs = loader.load()
    print(f"converted {len(outputs)} PDF(s) -> {args.output_dir}")


if __name__ == "__main__":
    main()
