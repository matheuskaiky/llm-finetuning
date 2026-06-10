#!/usr/bin/env python3
"""Download gazette PDFs listed (one URL per line) in a links file.

Usage:
    python scripts/download_diarios.py --links configs/diarios_links.txt \
        --out data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


def _filename_for(url: str, index: int) -> str:
    name = Path(urlparse(url).path).name
    return name if name.endswith(".pdf") else f"document_{index:04d}.pdf"


def download(links_file: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = [
        ln.strip()
        for ln in links_file.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    saved: list[Path] = []
    for i, url in enumerate(urls):
        dest = out_dir / _filename_for(url, i)
        if dest.exists():
            print(f"skip (exists): {dest.name}")
            saved.append(dest)
            continue
        try:
            with urlopen(url) as resp:  # noqa: S310 - trusted gazette URLs
                dest.write_bytes(resp.read())
            print(f"saved: {dest.name}")
            saved.append(dest)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"FAILED {url}: {exc}")
    print(f"\n{len(saved)}/{len(urls)} PDFs available in {out_dir}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", type=Path, required=True, help="file with URLs")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    download(args.links, args.out)


if __name__ == "__main__":
    main()
