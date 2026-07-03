"""Pure helpers for the source-anchored, deep-question variant of the Q2 dataset.

Same cleaning/parsing/dedup rules as ``sft_pairs.py`` (reused, not duplicated).
The only new pieces are: the deep-question prompt, the projection into an
SFT-ready record vs. a source-anchored (audit) record, and archiving of
previously generated outputs before a new run overwrites them. No ML stack;
unit-testable in isolation. Generation itself lives in
``scripts/build_sft_pairs_anchored.py``.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .sft_pairs import dedup_pairs, parse_qa_pairs


class ChatLLM(Protocol):
    def chat(self, messages: list[dict[str, str]], max_new_tokens: int) -> str: ...


class GenerationAborted(RuntimeError):
    """Raised when the engine fails on the first calls (config/load problem,
    not a one-off bad document)."""

# Belt-and-suspenders check: parse_qa_pairs/_coerce_pair (sft_pairs.py) only
# guarantee non-empty instruction/output. The prompt asks for deep questions,
# but the model can still ignore that, so shallow openers and near-empty
# answers are rejected here rather than trusted blindly.
_SHALLOW_PATTERN = re.compile(
    r"^(o que (e|é)|defina|define|o que significa|para que serve)\b",
    re.IGNORECASE,
)

DEEP_QA_SYSTEM_PROMPT = (
    "Você cria um par de instrução (pergunta e resposta) em português a partir "
    "de um trecho do material de um professor do Departamento de Computação, "
    "para treinar um modelo por instruction tuning.\n\n"
    "Gere EXATAMENTE 1 par pergunta-resposta PROFUNDO sobre o CONTEÚDO "
    "ESPECÍFICO do trecho abaixo. A resposta deve se basear apenas no trecho, "
    "sem inventar fatos.\n\n"
    "NÃO gere perguntas rasas de definição genérica, como (exemplos do que "
    "EVITAR):\n"
    "- \"O que é [conceito]?\"\n"
    "- \"Defina [termo].\"\n"
    "- \"O que significa [sigla]?\"\n"
    "- \"Para que serve [ferramenta]?\"\n\n"
    "Prefira perguntas que só podem ser respondidas por quem leu ESTE trecho "
    "específico, por exemplo (estilo, não copiar literalmente):\n"
    "- Uma comparação ou trade-off entre duas abordagens mencionadas no "
    "trecho.\n"
    "- O motivo (o \"por quê\") de uma escolha, passo ou resultado descrito.\n"
    "- Uma consequência ou o próximo passo de um procedimento/algoritmo do "
    "trecho.\n"
    "- Um detalhe concreto (um número, nome, exemplo, trecho de código) "
    "citado.\n"
    "- \"O que aconteceria se [condição do trecho] não fosse satisfeita?\"\n\n"
    "Regras:\n"
    "- A pergunta deve ser autônoma (sem \"segundo o texto\", \"no trecho "
    "acima\" etc.).\n"
    "- A resposta (\"output\") deve ser fundamentada SOMENTE no trecho, "
    "direta, completa e pronta para uso imediato como resposta final: sem "
    "frases de abertura (\"Claro,\", \"A resposta é:\") nem comentários sobre "
    "a própria pergunta.\n"
    "- O campo \"input\" é opcional e fica a seu critério: use-o só quando um "
    "contexto curto (ex.: um trecho de código ou dado citado no trecho) for "
    "necessário para a pergunta fazer sentido sozinha; caso contrário deixe "
    "\"\" vazio.\n"
    "- Se o trecho for raso demais para uma pergunta profunda (ex.: apenas um "
    "título ou lista de nomes), gere o melhor par possível mesmo assim, sem "
    "inventar conteúdo.\n"
    "- Responda APENAS com um array JSON válido de 1 elemento: "
    '[{"instruction": "...", "input": "", "output": "..."}]. Escape '
    "corretamente aspas, barras invertidas e quebras de linha dentro dos "
    "campos (\\n) para manter o JSON válido; nada fora do array (sem "
    "markdown, sem explicações)."
)


def is_deep_enough(pair: dict, min_output_chars: int = 20) -> bool:
    """Reject pairs that slipped past the prompt's own "no shallow questions" rule.

    Rejects: an instruction starting with a generic-definition opener (the exact
    patterns the prompt tells the model to avoid), an output too short to be a
    real grounded answer, or an output that just echoes the instruction.
    """
    instruction = pair.get("instruction", "").strip()
    output = pair.get("output", "").strip()
    if _SHALLOW_PATTERN.match(instruction):
        return False
    if len(output) < min_output_chars:
        return False
    if output.casefold() == instruction.casefold():
        return False
    return True


def build_user_message(nome: str, text: str) -> str:
    """Render the professor name and source excerpt into the user turn."""
    return f"Professor: {nome}\nTrecho:\n{text}\n\nJSON:"


def to_sft_record(pair: dict) -> dict:
    """Project a generated pair into the SFT-ready ``{instruction,input,output}``.

    Drops any source/traceability fields so the record matches the schema
    ``SftPairsLoader`` already expects (same path used by the training configs).
    """
    return {
        "instruction": pair["instruction"],
        "input": pair.get("input", ""),
        "output": pair["output"],
    }


def to_source_record(pair: dict, professor: str, doc_index: int, excerpt: str) -> dict:
    """Attach source/traceability metadata to a generated pair (audit dataset)."""
    return {
        "instruction": pair["instruction"],
        "input": pair.get("input", ""),
        "output": pair["output"],
        "professor": professor,
        "doc_index": doc_index,
        "source_excerpt": excerpt,
    }


def generate_pairs_for_records(
    llm: ChatLLM,
    records: list[dict],
    target: int,
    max_chars: int,
    max_new_tokens: int,
    on_progress: Callable[[dict], None] | None = None,
    abort_after_consecutive_errors: int = 5,
) -> tuple[list[dict], list[dict], dict]:
    """Generate one deep pair per record until ``target`` unique pairs collected.

    Handles every way the model can fail to produce a usable pair without
    crashing the run: non-JSON/malformed output (``parse_qa_pairs`` returns
    nothing), a shallow/too-short pair (``is_deep_enough`` rejects it), and an
    exception raised by ``llm.chat`` itself (caught and counted as an error, the
    record is just skipped). Only ``abort_after_consecutive_errors`` *consecutive*
    ``llm.chat`` exceptions raise ``GenerationAborted`` - that pattern means the
    engine itself is broken (bad model path, OOM, ...), not a one-off bad
    document, so it is not worth burning the rest of the job's time budget.

    Returns ``(sft_pairs, source_pairs, stats)`` where ``stats`` has
    ``calls``, ``malformed``, ``shallow`` and ``errors`` counters.
    """
    collected: list[dict] = []
    sources: list[dict] = []
    stats = {"calls": 0, "malformed": 0, "shallow": 0, "errors": 0}
    consecutive_errors = 0

    for doc_index, rec in enumerate(records):
        if len(dedup_pairs(collected)) >= target:
            break
        text = rec.get("text", "")[:max_chars]
        nome = rec.get("nome_professor", "?")
        stats["calls"] += 1

        try:
            raw = llm.chat(
                [
                    {"role": "system", "content": DEEP_QA_SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(nome, text)},
                ],
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - one bad doc must not kill the job
            stats["errors"] += 1
            consecutive_errors += 1
            if consecutive_errors >= abort_after_consecutive_errors:
                raise GenerationAborted(
                    f"llm.chat failed {consecutive_errors} times in a row "
                    f"(last: {exc!r}); aborting instead of burning the run."
                ) from exc
            if on_progress:
                on_progress({**stats, "doc_index": doc_index, "event": "error", "error": repr(exc)})
            continue

        consecutive_errors = 0
        pairs = parse_qa_pairs(raw)[:1]  # exactly one deep pair per document
        if not pairs:
            stats["malformed"] += 1
        for p in pairs:
            if not is_deep_enough(p):
                stats["shallow"] += 1
                continue
            collected.append(p)
            sources.append(to_source_record(p, professor=nome, doc_index=doc_index, excerpt=text))

        if on_progress:
            on_progress({**stats, "doc_index": doc_index, "unique": len(dedup_pairs(collected))})

    unique = dedup_pairs(collected)[:target]
    unique_keys = {(p["instruction"], p.get("input", "")) for p in unique}
    unique_sources = [
        s for s in sources if (s["instruction"], s.get("input", "")) in unique_keys
    ][: len(unique)]
    return unique, unique_sources, stats


def archive_existing_outputs(paths: list[Path], old_dir: Path) -> None:
    """Move any of ``paths`` that already exist into ``old_dir`` before a rerun.

    Idempotent: skips paths that do not exist, and overwrites a previous archive
    of the same filename in ``old_dir`` (last run wins). Creates ``old_dir`` on
    demand.
    """
    existing = [p for p in paths if p.exists()]
    if not existing:
        return
    old_dir.mkdir(parents=True, exist_ok=True)
    for p in existing:
        shutil.move(str(p), str(old_dir / p.name))
