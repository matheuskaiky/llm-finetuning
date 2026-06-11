"""Tests for the docente (SIGAA) extraction and exact-dedup logic.

These exercise the pure logic (triage, path/name metadata, text fingerprint,
deduplication) plus a small end-to-end ``load`` over plain-text fixtures, all
runnable without the ML stack.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_finetuning.data.docente_extractor import (
    DocenteExtractor,
    classify_file,
    deduplicate,
    parse_path_metadata,
    parse_sigaa_id,
    sanitize_text,
    text_fingerprint,
)


def test_sanitize_text_drops_surrogates_keeps_accents() -> None:
    dirty = "Avalia\udce7\udce3o \ud800final acentuação"
    clean = sanitize_text(dirty)
    # Lone surrogates are removed; valid UTF-8 (including accents) survives.
    assert clean.encode("utf-8")  # must be encodable, no exception
    assert "acentuação" in clean
    assert "\ud800" not in clean


def test_classify_file_buckets() -> None:
    assert classify_file(Path("a/b/lista.pdf")) == "text"
    assert classify_file(Path("a/b/slides.PPTX")) == "text"
    assert classify_file(Path("a/b/Main.java")) == "code"
    assert classify_file(Path("a/b/solver.py")) == "code"
    assert classify_file(Path("a/b/app.exe")) == "noise"
    assert classify_file(Path("a/b/mesh.obj")) == "noise"


def test_classify_file_noise_dirs_override_extension() -> None:
    # A .py inside a vendored environment is noise, not code.
    assert classify_file(Path("prof/.venv/lib/site-packages/numpy/x.py")) == "noise"
    assert classify_file(Path("prof/__MACOSX/lista.pdf")) == "noise"
    assert classify_file(Path("prof/_archives/grupo8.zip")) == "noise"
    assert classify_file(Path("prof/2024/._lista.pdf")) == "noise"


def test_parse_sigaa_id_strips_numeric_prefix() -> None:
    assert parse_sigaa_id("6228726-Trabalho Estoque 1.pdf") == (
        "6228726",
        "Trabalho Estoque 1.pdf",
    )
    assert parse_sigaa_id("lista_sem_id.pdf") == (None, "lista_sem_id.pdf")


def test_parse_path_metadata_from_tree() -> None:
    root = Path("/data/docentesDC-sigaa")
    p = root / "Joao Silva" / "2024" / "03" / "05" / "6228726-Lista.pdf"
    meta = parse_path_metadata(root, p)
    assert meta["professor"] == "Joao Silva"
    assert meta["year"] == 2024
    assert meta["month"] == 3
    assert meta["day"] == 5
    assert meta["date"] == "2024-03-05"


def test_text_fingerprint_ignores_case_and_whitespace() -> None:
    a = text_fingerprint("Lista   de\n\nExercicios")
    b = text_fingerprint("lista de exercicios")
    assert a == b
    assert a != text_fingerprint("outro conteudo")


def test_deduplicate_keeps_most_recent_canonical() -> None:
    records = [
        {
            "professor": "Ana",
            "date": "2022-03-01",
            "text_sha1": "h1",
            "content_md5": "m1",
            "source_path": "Ana/2022/03/01/lista.pdf",
        },
        {
            "professor": "Ana",
            "date": "2024-03-01",
            "text_sha1": "h1",
            "content_md5": "m2",
            "source_path": "Ana/2024/03/01/lista.pdf",
        },
    ]
    out = deduplicate(records)
    assert len(out) == 1
    canon = out[0]
    assert canon["date"] == "2024-03-01"
    assert canon["dup_count"] == 2
    assert canon["duplicated_dates"] == ["2022-03-01", "2024-03-01"]


def test_deduplicate_marks_cross_professor_shared() -> None:
    records = [
        {
            "professor": "Ana",
            "date": "2023-01-01",
            "text_sha1": "shared",
            "content_md5": "x",
            "source_path": "Ana/2023/01/01/padrao.pdf",
        },
        {
            "professor": "Bruno",
            "date": "2023-01-01",
            "text_sha1": "shared",
            "content_md5": "y",
            "source_path": "Bruno/2023/01/01/padrao.pdf",
        },
    ]
    out = deduplicate(records)
    assert len(out) == 1
    assert out[0]["shared_with_professors"] == ["Ana", "Bruno"]


def test_deduplicate_groups_by_md5_when_text_missing() -> None:
    records = [
        {
            "professor": "Ana",
            "date": "2021-01-01",
            "text_sha1": "",
            "content_md5": "same",
            "source_path": "Ana/2021/01/01/scan.pdf",
        },
        {
            "professor": "Ana",
            "date": "2022-01-01",
            "text_sha1": "",
            "content_md5": "same",
            "source_path": "Ana/2022/01/01/scan.pdf",
        },
    ]
    out = deduplicate(records)
    assert len(out) == 1
    assert out[0]["date"] == "2022-01-01"


def test_write_jsonl_survives_surrogate_in_any_field(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    extractor = DocenteExtractor(input_dir=tmp_path, output_path=out)
    # Surrogate lands in a path-derived field, not in ``text``.
    record = {"professor": "Jo\udce3o", "text": "ok", "source_path": "a/b\udce7.pdf"}
    extractor._write_jsonl([record])
    line = out.read_text(encoding="utf-8").strip()
    loaded = json.loads(line)
    assert loaded["text"] == "ok"
    assert "\udce3" not in loaded["professor"]


def _write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_load_end_to_end_excludes_code_and_dedups(tmp_path: Path) -> None:
    root = tmp_path / "docentesDC-sigaa"
    # Same text reused across two years by one professor (should dedup to 1).
    _write(root / "Ana" / "2022" / "03" / "01" / "10-Lista.txt", "Conteudo A")
    _write(root / "Ana" / "2024" / "03" / "01" / "20-Lista.txt", "conteudo   a")
    # A distinct text document.
    _write(root / "Ana" / "2024" / "03" / "01" / "21-Prova.txt", "Conteudo B")
    # Code and noise that must not appear in the factual corpus.
    _write(root / "Ana" / "2024" / "03" / "01" / "Main.java", "class Main {}")
    _write(root / "Ana" / "_archives" / "grupo.txt", "ignore me")

    out_path = tmp_path / "docentes.jsonl"
    extractor = DocenteExtractor(input_dir=root, output_path=out_path)
    records = extractor.load()

    texts = {r["text"] for r in records}
    # The reused "Lista" collapses to its most recent version (2024, lowercased).
    assert "conteudo a" in texts
    assert "Conteudo B" in texts
    assert all("class Main" not in r["text"] for r in records)
    assert all("ignore me" not in r["text"] for r in records)
    # Two canonical docs: deduped Lista + Prova.
    assert len(records) == 2
    lista = next(r for r in records if r["text"] == "conteudo a")
    assert lista["dup_count"] == 2
    assert lista["date"] == "2024-03-01"

    # JSONL written, one object per line, matching the returned records.
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(records)
    assert json.loads(lines[0])["professor"] == "Ana"


def test_load_with_include_code_adds_code_subcorpus(tmp_path: Path) -> None:
    root = tmp_path / "docentesDC-sigaa"
    _write(root / "Ana" / "2024" / "03" / "01" / "10-Lista.txt", "Conteudo A")
    _write(root / "Ana" / "2024" / "03" / "01" / "Main.java", "class Main {}")

    extractor = DocenteExtractor(
        input_dir=root, output_path=tmp_path / "out.jsonl", include_code=True
    )
    records = extractor.load()
    filetypes = {r["filetype"] for r in records}
    assert "java" in filetypes
