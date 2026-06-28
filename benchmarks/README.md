# benchmarks/

Conjuntos de avaliação e resultados.

## Formatos aceitos

- **`.jsonl`** - um objeto JSON por linha. O avaliador intrínseco usa o campo
  `text`; quando ausente, concatena `instruction` / `input` / `output`.
- **`.txt`** - o arquivo inteiro é tratado como um único documento.

## Arquivos

- `sample_corpus.jsonl` - corpus mínimo (exemplos de diários) para o baseline (#4).
- `pre_treino/` - benchmark da fase de pré-treino (Q1, diários). Avaliação
  antes/depois do pré-treino. Ver `pre_treino/README.md`.
- `pos_treino/` - fase de pós-treino (Q2 SFT e Q3 LoRA/QLoRA, docentes). Os pares de
  treino, held-out e recall são derivados do `docentesDC` por
  `scripts/build_sft_pairs.py` (não versionados); as saídas antes/depois ficam em
  `pos_treino/results/`. Ver `pos_treino/README.md`.
- `rag/` - benchmark da Q5 (30 perguntas factual/multi-hop ancoradas no índice
  GraphRAG dos diários). Ver `rag/README.md`.
- `guardrails/` - benchmark da Q6 (prompts adversariais + casos de PII). Ver
  `guardrails/README.md`.
- `results/` - saídas de avaliação em JSON (geradas; não versionadas).

## Organização por fase de treino

A separação de alto nível segue as fases do ciclo de vida (slides): pré-treino
(Q1) e pós-treino (Q2/Q3). Dentro de cada fase, a avaliação é feita antes e depois
do treino daquela fase (`results/antes/`, `results/depois/`).

## Por questão

| Questão | Benchmark | Tamanho | Status |
|---------|-----------|---------|--------|
| Q1 | P&R sobre os diários (`pre_treino/diarios_qa.jsonl`) + held-out de texto | 33 | feito |
| Q2/Q3 | Pares do `docentesDC` (treino + held-out + recall), `pos_treino/` | 1.000 + 150 + 150 | feito |
| Q4 | Recall / held-out teacher-student (diários) | 100 / 100 | feito |
| Q5 | RAG (`rag/diarios_rag_30.jsonl`, factual + multi-hop) | 30 | feito |
| Q6 | Guardrails (`guardrails/`, adversarial + PII + benigno) | 30 + 15 | feito |
