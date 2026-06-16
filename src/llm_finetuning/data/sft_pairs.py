"""Pure helpers for building the Q2 SFT dataset from the docentesDC corpus.

Cleans the source records (exact dedup, drop garbled/short texts) and parses /
dedups the instruction-tuning pairs produced by the generator LLM. No ML stack;
unit-testable in isolation. The generation itself lives in
``scripts/build_sft_pairs.py``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from ..core.interfaces import DatasetLoader
from ..core.registry import DATASET_LOADERS

# Instruction template for SFT. Base models have no chat template, so a plain,
# explicit Portuguese template is used; the trainer masks the loss to the response.
PROMPT_NO_INPUT = "### Instrucao:\n{instruction}\n\n### Resposta:\n"
PROMPT_WITH_INPUT = (
    "### Instrucao:\n{instruction}\n\n### Entrada:\n{input}\n\n### Resposta:\n"
)

# Letters expected in Portuguese text (besides ASCII). Records dominated by
# letters outside this set are PDF-extraction garbage (wrong font encoding).
_PT_EXTRA = set("áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇºªnÑñ")


def weird_letter_ratio(text: str) -> float:
    """Fraction of letters that are neither ASCII nor common Portuguese letters.

    Near 0 for normal Portuguese or code (mostly ASCII); near 1 for text whose
    glyphs come from a broken PDF font encoding.
    """
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    weird = sum(1 for c in letters if not (c.isascii() or c in _PT_EXTRA))
    return weird / len(letters)


def is_clean_record(text: str, min_chars: int = 30, max_weird: float = 0.30) -> bool:
    """A source text is usable when it is long enough and not glyph garbage."""
    t = (text or "").strip()
    if len(t) < min_chars:
        return False
    return weird_letter_ratio(t) <= max_weird


def clean_source_records(
    records: list[dict],
    text_key: str = "text",
    min_chars: int = 30,
    max_weird: float = 0.30,
) -> list[dict]:
    """Drop exact-duplicate, too-short and glyph-garbage source records.

    Keeps the first occurrence of each distinct text. Order is preserved.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        text = r.get(text_key, "")
        if not is_clean_record(text, min_chars, max_weird):
            continue
        key = text.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _coerce_pair(obj: dict) -> dict | None:
    """Validate one generated object into ``{instruction, input, output}``."""
    if not isinstance(obj, dict):
        return None
    instruction = str(obj.get("instruction", "")).strip()
    output = str(obj.get("output", "")).strip()
    input_ = str(obj.get("input", "")).strip()
    if not instruction or not output:
        return None
    return {"instruction": instruction, "input": input_, "output": output}


def parse_qa_pairs(raw: str) -> list[dict]:
    """Parse the generator output into a list of ``{instruction, input, output}``.

    Accepts a JSON array of objects (preferred) or, as a fallback, a sequence of
    individual JSON objects. Malformed items are skipped.
    """
    if not raw:
        return []
    pairs: list[dict] = []
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            if isinstance(data, list):
                for obj in data:
                    p = _coerce_pair(obj)
                    if p:
                        pairs.append(p)
                return pairs
        except (json.JSONDecodeError, ValueError):
            pass
    # Fallback: scan individual {...} objects.
    for m in re.finditer(r"\{[^{}]*\}", raw, flags=re.DOTALL):
        try:
            p = _coerce_pair(json.loads(m.group(0)))
        except (json.JSONDecodeError, ValueError):
            p = None
        if p:
            pairs.append(p)
    return pairs


def _norm_question(instruction: str, input_: str = "") -> str:
    instruction = re.sub(r"\s+", " ", instruction).strip()
    input_ = re.sub(r"\s+", " ", input_).strip()
    return f"{instruction}␟{input_}".casefold()


def dedup_pairs(pairs: list[dict]) -> list[dict]:
    """Drop pairs with a duplicate (instruction, input), keeping the first."""
    seen: set[str] = set()
    out: list[dict] = []
    for p in pairs:
        key = _norm_question(p.get("instruction", ""), p.get("input", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def subtract_pairs(pairs: list[dict], exclude: list[dict]) -> list[dict]:
    """Drop pairs whose (instruction, input) matches one in ``exclude``.

    Used to build an in-domain recall set from training source texts without
    leaking the exact training questions.
    """
    blocked = {
        _norm_question(p.get("instruction", ""), p.get("input", "")) for p in exclude
    }
    return [
        p
        for p in pairs
        if _norm_question(p.get("instruction", ""), p.get("input", "")) not in blocked
    ]


def build_prompt(instruction: str, input_: str = "") -> str:
    """Render the instruction (and optional input) into the SFT prompt prefix."""
    instruction = (instruction or "").strip()
    input_ = (input_ or "").strip()
    if input_:
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=input_)
    return PROMPT_NO_INPUT.format(instruction=instruction)


def build_input_and_labels(
    prompt_ids: list[int],
    output_ids: list[int],
    eos_id: int | None = None,
    max_length: int | None = None,
) -> tuple[list[int], list[int]]:
    """Concatenate prompt + response tokens and mask the prompt in the labels.

    Returns ``(input_ids, labels)`` where the prompt positions are ``-100`` (loss
    ignored) and only the response (plus EOS) carries a label. Both lists are
    truncated to ``max_length`` when given.
    """
    tail = list(output_ids) + ([eos_id] if eos_id is not None else [])
    input_ids = list(prompt_ids) + tail
    labels = [-100] * len(prompt_ids) + tail
    if max_length is not None:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return input_ids, labels


@DATASET_LOADERS.register("sft_pairs")
class SftPairsLoader(DatasetLoader):
    """Load instruction pairs (``{instruction, input?, output}``) from a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[dict]:
        rows: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
