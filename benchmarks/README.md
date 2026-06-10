# benchmarks/

Conjuntos de avaliação e resultados.

## Formatos aceitos

- **`.jsonl`** - um objeto JSON por linha. O avaliador intrínseco usa o campo
  `text`; quando ausente, concatena `instruction` / `input` / `output`.
- **`.txt`** - o arquivo inteiro é tratado como um único documento.

## Arquivos

- `sample_corpus.jsonl` - corpus mínimo (exemplos de diários) para o baseline (#4).
- `results/` - saídas de avaliação em JSON (geradas; não versionadas).

## Por questão (a construir)

| Questão | Benchmark | Tamanho |
|---------|-----------|---------|
| Q1 | P&R sobre os diários | >= 25 |
| Q4 | Avaliação teacher/student | 100 |
| Q5 | RAG | 30 |
| Q6 | Guardrails (adversarial + legítimo) | 30 |
