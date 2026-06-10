"""Plain-text corpus loader and token-block helper for causal-LM pretraining."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..core.interfaces import DatasetLoader
from ..core.registry import DATASET_LOADERS


def chunk_token_ids(
    token_ids: Sequence[int], block_size: int, drop_remainder: bool = True
) -> list[list[int]]:
    """Split a flat token-id sequence into fixed-size blocks.

    Args:
        token_ids: Concatenated token ids.
        block_size: Length of each block.
        drop_remainder: If True, discard a trailing block shorter than
            ``block_size``; otherwise keep it.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    n_full = len(token_ids) // block_size
    blocks = [
        list(token_ids[i * block_size : (i + 1) * block_size]) for i in range(n_full)
    ]
    remainder = token_ids[n_full * block_size :]
    if remainder and not drop_remainder:
        blocks.append(list(remainder))
    return blocks


@DATASET_LOADERS.register("text_corpus")
class TextCorpusLoader(DatasetLoader):
    """Loads every text file under a directory as one document per file.

    Args:
        input_dir: Directory scanned recursively.
        pattern: Glob pattern of files to read.
        encoding: Text encoding.
    """

    def __init__(
        self, input_dir: str | Path, pattern: str = "*.txt", encoding: str = "utf-8"
    ) -> None:
        self.input_dir = Path(input_dir)
        self.pattern = pattern
        self.encoding = encoding

    def load(self) -> list[str]:
        files = sorted(self.input_dir.rglob(self.pattern))
        return [f.read_text(encoding=self.encoding) for f in files]
