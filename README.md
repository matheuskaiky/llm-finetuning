# llm-finetuning

End-to-end Large Language Model (LLM) lifecycle project built for the course
**Tópicos em IA** (UFPI-CCN-DC, Prof. Raimundo Moura - 2026.1).

The repository implements the full training and serving pipeline of an LLM
specialized in public-sector data from Piauí (municipal official gazettes -
*diários das prefeituras*) and university faculty data (*docentesDC*), covering
six fronts:

1. **Continued pre-training** of a base LLM + quality evaluation (before/after).
2. **Supervised Fine-Tuning (SFT)** from 1,000+ instruction/response pairs.
3. **LoRA / QLoRA** parameter-efficient fine-tuning.
4. **Knowledge distillation** (teacher -> student).
5. **RAG** (Retrieval-Augmented Generation) assistant.
6. **Guardrails** safety layer.

## Tech stack

Python 3.12 - PyTorch - Hugging Face (`transformers`, `datasets`, `accelerate`,
`peft`, `trl`, `bitsandbytes`) - RAG (`langchain`/`llama-index` + `faiss`/`chromadb`
+ `sentence-transformers`) - Guardrails (`nemoguardrails`/`guardrails-ai`) -
config-driven design (YAML) for environment-agnostic execution (local GPU <-> cloud).

## Project documentation

| File | Purpose |
|------|---------|
| [`CONTEXT.md`](CONTEXT.md) | Visão de produto, escopo macro, cronograma e formato da entrega final. |
| [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) | Contexto técnico: arquitetura, árvore de diretórios e contratos. |

## Repository layout

```
src/          Source code (core abstractions + the six task modules)
configs/      YAML configuration (models, methods, environment) - OCP-friendly
data/         Datasets (raw / processed) - not versioned
benchmarks/   Evaluation question sets
notebooks/    Exploratory experiments
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

## Download the model and dataset

Weights and datasets are not versioned (see `.gitignore`); each person downloads
them locally.

### Base model + gazette corpus (Hugging Face)

IDs come from `.env` (`BASE_MODEL_ID`, `DATASET_ID`).

```bash
# Base model (Qwen/Qwen3.5-9B, ~18 GB) into models/ and
# gazette corpus (gutoportelaa/dom-pi-corpus-2025) into data/raw/
uv run python scripts/download_assets.py --all

# Or one at a time
uv run python scripts/download_assets.py --model
uv run python scripts/download_assets.py --dataset
```

### Docente dataset (docentesDC / SIGAA, Google Drive)

Source folder (one `.zip` per group, nested as
`TIA-Dados_Professores/grupoN/grupoN.zip`, each grouping files per professor):

<https://drive.google.com/drive/folders/1aDoEszVYDH1-nNoskLSMCfNLN_cjV16K>

Download it through the browser (the Takeout direct links are session-bound and
do not work with `curl`/`wget` from a server) and place the `.zip` parts in
`data/raw/docentesDC-sigaa/`. If the folder is shared as "anyone with the link",
`gdown` also works:

```bash
uvx gdown --folder "https://drive.google.com/drive/folders/1aDoEszVYDH1-nNoskLSMCfNLN_cjV16K" -O data/raw/docentesDC-sigaa
```

Then extract, flattening the `grupoN` layer so the tree is `professor/year/...`:

```bash
# (see the extraction step used in data/raw/docentesDC-sigaa: outer parts ->
# inner grupoN.zip -> professor folders at the top level; originals kept in _archives/)
```

## Branch policy

Never commit to `main`. Work on other branches (`dev` or feature branches) and
merge into `main` through pull requests. Pushes use the author's own GitHub
account.

## Delivery milestones

- **2026-06-11** - First partial presentation (setup + MVP).
- **2026-06-25** - Second partial presentation (core features).
- **2026-07-07** - Final presentation and delivery (integration, polish, bug fixing).

See [`CONTEXT.md`](CONTEXT.md) for the full schedule and scope breakdown.
