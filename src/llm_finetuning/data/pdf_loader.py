"""PDF-to-text loader for the municipal gazettes (``diariosPrefeituras``).

Extracts text from PDFs with ``pypdf`` (imported lazily), normalizes whitespace,
and drops boilerplate lines that repeat across most pages (headers/footers).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ..core.interfaces import DatasetLoader
from ..core.registry import DATASET_LOADERS

_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Collapse horizontal whitespace and runs of blank lines, strip line ends."""
    lines = [_WHITESPACE.sub(" ", line).rstrip() for line in text.splitlines()]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _drop_repeated_lines(pages: list[str], threshold: float) -> list[str]:
    """Remove lines that appear on at least ``threshold`` fraction of pages."""
    if len(pages) < 3 or threshold >= 1.0:
        return pages
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({ln.strip() for ln in page.splitlines() if ln.strip()})
    cutoff = max(2, int(threshold * len(pages)))
    boilerplate = {ln for ln, c in counts.items() if c >= cutoff}
    if not boilerplate:
        return pages
    cleaned = []
    for page in pages:
        kept = [ln for ln in page.splitlines() if ln.strip() not in boilerplate]
        cleaned.append("\n".join(kept))
    return cleaned


@DATASET_LOADERS.register("pdf_to_text")
class PdfToTextLoader(DatasetLoader):
    """Converts a directory of PDFs into normalized ``.txt`` files.

    Args:
        input_dir: Directory scanned recursively for ``*.pdf``.
        output_dir: Where ``.txt`` files are written (mirrors stem names).
        boilerplate_threshold: Fraction of pages a line must appear on to be
            treated as header/footer boilerplate and removed (1.0 disables it).
    """

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        boilerplate_threshold: float = 0.6,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.boilerplate_threshold = boilerplate_threshold

    def extract_pages(self, pdf_path: str | Path) -> list[str]:
        """Return the per-page text of a single PDF."""
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return [page.extract_text() or "" for page in reader.pages]

    def extract_text(self, pdf_path: str | Path) -> str:
        """Return the cleaned full text of a single PDF."""
        pages = self.extract_pages(pdf_path)
        pages = _drop_repeated_lines(pages, self.boilerplate_threshold)
        return normalize_text("\n\n".join(pages))

    def load(self) -> list[Path]:
        """Convert every PDF under ``input_dir`` and return the output paths."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        for pdf_path in sorted(self.input_dir.rglob("*.pdf")):
            text = self.extract_text(pdf_path)
            out_path = self.output_dir / f"{pdf_path.stem}.txt"
            out_path.write_text(text, encoding="utf-8")
            outputs.append(out_path)
        return outputs
