# llm-finetuning

End-to-end Large Language Model (LLM) lifecycle project built for the course
**Tópicos em IA** (UFPI-CCN-DC, Prof. Raimundo Moura - 2026.1).

The repository implements the full lifecycle of an LLM specialized in public-sector
data from Piauí: from continued pre-training on municipal official gazettes, through
supervised and parameter-efficient fine-tuning, knowledge distillation, a
Retrieval-Augmented Generation (RAG) assistant, and a safety guardrails layer. Every
stage is driven by YAML configuration so models, methods and the execution environment
(local GPU or cloud) can be swapped without touching code (Open-Closed design).

## The six fronts (Q1-Q6)

| # | Front | Data it uses | Headline result |
|---|-------|--------------|-----------------|
| Q1 | Continued pre-training of a base model + before/after evaluation | Gazettes corpus (`dom-pi-corpus-2025`) | Fine-tuned base beats off-the-shelf instruct; held-out PPL down to 5.49 |
| Q2 | Supervised Fine-Tuning (SFT) from 1,000+ instruction/response pairs | Faculty data (`docentesDC`) | SFT lowers response PPL across the board; Q1+SFT compounds on Qwen |
| Q3 | LoRA / QLoRA parameter-efficient fine-tuning | Faculty data (`docentesDC`) | LoRA matches/beats full SFT training ~1.7% of the parameters |
| Q4 | Knowledge distillation (teacher -> student) | Gazettes (synthetic Q&A from a teacher) | Knowledge transfers; a 0.5B student becomes a strong RAG engine |
| Q5 | RAG assistant (Standard / Agentic / GraphRAG) | Gazettes corpus (`dom-pi-corpus-2025`) | Retrieval is the main gain (baseline ~1.1 -> RAG ~2.7-3.9) |
| Q6 | Composable guardrails safety layer | Synthetic adversarial + PII benchmark | 0 -> 100% block/mask of attacks and PII, no false positives on benign |

Consolidated numbers, tables and the cross-cutting analysis live in
[`results/README.md`](results/README.md) and the notebook
[`notebooks/graficos_resultados.ipynb`](notebooks/graficos_resultados.ipynb).

## Datasets

Two public Hugging Face datasets back the whole project. Neither is versioned in git
(see `.gitignore`); each person downloads them locally (instructions below).

### Diários Oficiais dos Municípios do Piauí 2025 (gazettes)

- Hub id: `gutoportelaa/dom-pi-corpus-2025` (set via `DATASET_ID` in `.env`).
- Content: municipal official gazettes (*diários oficiais*) of Piauí for 2025. The
  raw text lives in the `texto` column; the repo preprocesses it to plain `.txt`
  documents (about 2,000 documents in the working snapshot).
- Domain: dense public-sector text (acts, appointments, bids/licitações, budgets),
  formulaic and repetitive, which is exactly the specialization target.
- Where it is used:
  - **Q1 continued pre-training** - the causal-LM corpus the base model is trained on
    (chunked into token blocks). A balanced variant (pruned licitações) is kept as an
    ablation.
  - **Q4 distillation** - the teacher generates synthetic Q&A anchored on the
    gazettes; the in-domain recall benchmark for teacher/student is built from them.
  - **Q5 RAG** - the knowledge base that is chunked, embedded and indexed (FAISS +
    graph), and the source for the 30-question retrieval benchmark.

### docentesDC (university faculty data)

- Hub id: `vickminari/docentesDC` (official, pre-processed dataset, 13,762 records,
  fields `text` and `nome_professor`). It supersedes an earlier SIGAA scrape this repo
  used to extract itself; the old `DocenteExtractor` pipeline was removed.
- Where it is used:
  - **Q2 SFT and Q3 LoRA/QLoRA** - source of the 1,000+ `{instruction, input?, output}`
    pairs the models are fine-tuned on, plus disjoint held-out and recall sets for
    before/after evaluation.
  - **Q1 forgetting diagnostic** - serves as an out-of-distribution probe to measure
    catastrophic forgetting after the gazette pre-training.

The Q6 guardrails benchmark is a small hand-built set of adversarial prompts and PII
cases, not derived from either corpus.

Evaluation note: the current benchmark question/answer references were generated with
AI assistance (the judge is also an LLM, fixed Qwen3-8B on a 0-5 scale). A
hand-written, no-AI gold set is planned as an independent reference.

## Architecture

Core abstractions in `src/llm_finetuning/` (model providers, dataset loaders,
trainers, metrics, evaluators) are resolved from registries, and each experiment is a
YAML file in `configs/`. New models, training methods or metrics are added by
extension (a new class plus a config), not by editing the core. Scripts in `scripts/`
are thin CLI wrappers that read a config and dispatch into `src/`. See
[`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) for the full contracts and directory tree,
and [`docs/IMPLEMENTACAO_QUESTOES.md`](docs/IMPLEMENTACAO_QUESTOES.md) for which pieces
each question uses.

## Tech stack

Python 3.12 - PyTorch - Hugging Face (`transformers`, `datasets`, `accelerate`,
`peft`, `trl`, `bitsandbytes`) - RAG (`langchain` + `faiss` + `sentence-transformers`,
graph via `networkx`) - guardrails as a composable rule-based layer (no external
guardrails framework) - config-driven (YAML) for environment-agnostic execution.

Hardware context: experiments run on 2x NVIDIA L4 (22 GB each). Full fine-tuning of 4B
and a clean 31B RAG engine do not fit and are documented as hardware limits, not
quality results.

## Repository layout

```
src/          Source code (core abstractions + the six task modules)
configs/      YAML configuration (models, methods, environment) - OCP-friendly
data/         Datasets (raw / processed) - not versioned
benchmarks/   Evaluation question sets (pre_treino / rag / guardrails)
results/      Consolidated result CSVs + results/README.md ledger
notebooks/    Result analysis (graficos_resultados.ipynb)
scripts/      CLI entrypoints (train / eval / rag / ...)
tests/        Test suite (pytest)
docs/         Additional documentation
```

## Quickstart

The environment is managed with [uv](https://docs.astral.sh/uv/). Nothing is
installed into the global/conda Python.

```bash
# 1. Create the isolated .venv and install dependencies from uv.lock
uv sync

# 2. Run the test suite
uv run pytest

# 3. Copy the env template and fill in values if needed (e.g. HF_TOKEN)
cp .env.example .env
```

## Download the model and datasets

Weights and datasets are not versioned; each person downloads them locally. IDs come
from `.env` (`BASE_MODEL_ID`, `DATASET_ID`).

```bash
# Base model (BASE_MODEL_ID, default Qwen/Qwen3-8B) into models/ and the
# gazette corpus (gutoportelaa/dom-pi-corpus-2025) into data/raw/
uv run python scripts/download_assets.py --all

# Or one at a time
uv run python scripts/download_assets.py --model
uv run python scripts/download_assets.py --dataset
```

The faculty dataset (`vickminari/docentesDC`) is fetched from the Hub into
`data/raw/docentesDC/` (`docentesDC.jsonl`, `docentesDC.parquet`):

```bash
.venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id="vickminari/docentesDC", repo_type="dataset",
                  local_dir="data/raw/docentesDC", token=os.environ.get("HF_TOKEN"))
PY
```

## Project documentation

| File | Purpose |
|------|---------|
| [`CONTEXT.md`](CONTEXT.md) | Visão de produto, escopo macro, cronograma e formato da entrega. |
| [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) | Contexto técnico: arquitetura, árvore de diretórios e contratos. |
| [`docs/IMPLEMENTACAO_QUESTOES.md`](docs/IMPLEMENTACAO_QUESTOES.md) | Quais módulos, configs e scripts cada questão usa, e as decisões. |
| [`docs/PERGUNTAS.md`](docs/PERGUNTAS.md) | Quantidades e arquivos de dados de fato usados em cada questão. |
| [`docs/PROMPTS.md`](docs/PROMPTS.md) | Catálogo dos prompts de LLM (juiz, gerador, crítico, extrator). |
| [`results/README.md`](results/README.md) | Ledger consolidado dos resultados Q1-Q6 e síntese transversal. |

## Branch policy

Never commit directly to `main`. Work on other branches (`dev` or feature branches)
and merge into `main` through pull requests. Pushes use the author's own GitHub
account.

## Delivery milestones

- **2026-06-11** - First partial presentation (setup + MVP).
- **2026-06-25** - Second partial presentation (core features).
- **2026-07-07** - Final presentation and delivery (integration, polish, bug fixing).

See [`CONTEXT.md`](CONTEXT.md) for the full schedule and scope breakdown.
