# Resultados consolidados (ledger geral)

Registro acumulado de todas as execuções de treino/avaliação do projeto, para o
relatório e para comparar modelos e tamanhos sem perder corridas anteriores.
Nenhum resultado é descartado: cada corrida vira uma linha aqui, mesmo as
substituídas por modelos maiores depois.

- `runs.csv` - fonte de dados (uma linha por par modelo x conjunto de avaliação).
- Esta tabela - leitura humana da mesma informação.

As métricas brutas por corrida continuam também em
`benchmarks/<fase>/results/<antes|depois>/*.json` (git-ignored); este ledger é o
índice versionado que sobrevive a elas.

## Q1 - pré-treino contínuo (full-parameter)

Avaliação antes/depois do pré-treino sobre os diários. Perplexidade e entropia
cruzada: menor é melhor. Acurácia de token: maior é melhor.

| Data | Modelo | Tam. | Variante | Método | Conjunto aval. | PPL antes | PPL depois | CE antes | CE depois | TokAcc antes | TokAcc depois | GPUs |
|------|--------|------|----------|--------|----------------|-----------|------------|----------|-----------|--------------|---------------|------|
| 2026-06-11 | Qwen3-0.6B-Base | 0.6B | base | full param | held-out (150 docs, disjunto) | 11.467 | 6.884 | 2.439 | 1.929 | 0.524 | 0.603 | 1x L4 |
| 2026-06-11 | Qwen3-0.6B-Base | 0.6B | base | full param | diarios_qa (33 P&R) | 11.582 | 10.128 | 2.449 | 2.315 | 0.503 | 0.521 | 1x L4 |
| 2026-06-12 | Qwen3-1.7B-Base | 1.7B | base | full param | held-out (150 docs, disjunto) | 8.587 | 5.732 | 2.150 | 1.746 | 0.564 | 0.627 | 1x L4 |
| 2026-06-12 | Qwen3-1.7B-Base | 1.7B | base | full param | diarios_qa (33 P&R) | 7.708 | 7.068 | 2.042 | 1.956 | 0.547 | 0.552 | 1x L4 |

O conjunto held-out (texto de diário inédito) mostra o efeito maior, por ser texto
do mesmo domínio não visto no treino; as P&R conceituais movem menos por serem
genéricas. A acurácia de token é otimista no texto formulaico dos diários, então a
perplexidade é a métrica mais informativa.

Escada de tamanho (held-out, perplexidade depois): o modelo maior parte de uma base
melhor (antes: 11.47 no 0.6B, 8.59 no 1.7B) e chega mais baixo depois (6.88 no
0.6B, 5.73 no 1.7B). Ou seja, o pré-treino contínuo ajuda nos dois tamanhos e o
ganho absoluto se mantém ao escalar. O 4B (próxima rung) está pronto, mas depende da
correção do multi-GPU pela infra (ver `docs/SUPORTE_INFRA_MULTIGPU.md`).

## Convenção de colunas (runs.csv)

`date, question, model, params, variant (base|instruct|vlm), modality (text|vlm),
method, dataset, eval_set, ppl_before, ppl_after, ce_before, ce_after,
tokacc_before, tokacc_after, train_loss, blocks, gpus, config, notes`.
