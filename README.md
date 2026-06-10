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
| [`CONTEXT.md`](CONTEXT.md) | Visão de produto, escopo macro e cronograma de entregas. |
| [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) | Contexto técnico: arquitetura, árvore de diretórios e contratos. |
| [`tarefa.md`](tarefa.md) | Enunciado original da atividade. |

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

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies (editable)
pip install -e .

# 3. Run the test suite
pytest
```

## Delivery milestones

- **2026-06-11** - First partial presentation (setup + MVP).
- **2026-06-25** - Second partial presentation (core features).
- **2026-07-07** - Final presentation and delivery (integration, polish, bug fixing).

See [`CONTEXT.md`](CONTEXT.md) for the full schedule and scope breakdown.
