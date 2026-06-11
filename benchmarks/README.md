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
- `pos_treino/` - reservado para a fase de pós-treino (Q2/Q3, docentes); a
  construir. Não faz parte da Q1.
- `results/` - saídas de avaliação em JSON (geradas; não versionadas).

## Organização por fase de treino

A separação de alto nível segue as fases do ciclo de vida (slides): pré-treino
(Q1) e pós-treino (Q2/Q3). Dentro de cada fase, a avaliação é feita antes e depois
do treino daquela fase (`results/antes/`, `results/depois/`).

## Por questão

| Questão | Benchmark | Tamanho | Status |
|---------|-----------|---------|--------|
| Q1 | P&R sobre os diários (`pre_treino/`) | >= 25 | criado |
| Q2/Q3 | P&R sobre os docentes (`pos_treino/`) | a definir | a construir |
| Q4 | Avaliação teacher/student | 100 | a construir |
| Q5 | RAG | 30 | a construir |
| Q6 | Guardrails (adversarial + legítimo) | 30 | a construir |
