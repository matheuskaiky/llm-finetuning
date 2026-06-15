# Balanceamento de licitações no corpus da Q1

Documenta como foi construído o corpus balanceado dos diários para a Q1
(pré-treino contínuo) e o held-out correspondente. O objetivo é testar se podar
documentos de licitação (repetitivos) ajuda ou atrapalha o modelo, espelhando o
estudo de licitações feito no RAG (Q5), mas agora no corpus de treino.

## Princípio

Mexer **somente** nas licitações: manter todos os documentos não-licitação e
reduzir só a classe licitação. O original é preservado; o balanceado é um diretório
novo de symlinks, sem duplicar nem mover arquivos.

## Como uma licitação é detectada

`llm_finetuning.rag.doc_select.is_licitacao` marca um documento como licitação
quando ele contém pelo menos 2 marcadores distintos (`min_hits=2`) de uma lista de
termos de compras públicas (licitação, pregão, edital, homologação, adjudicação,
dispensa, inexigibilidade, registro de preço, ata de registro, etc.). Função pura,
testada em `tests/test_doc_select.py`.

## Redução (downsample) só das licitações

`doc_select.downsample_licitacao(docs, keep_fraction=0.5, seed=42)` mantém todos os
não-licitação e sorteia uma fração `keep_fraction` das licitações (determinístico
pela seed). Nada é feito nos demais documentos.

## Corpus de treino

Construído por `scripts/build_balanced_corpus.py`:

```sh
python scripts/build_balanced_corpus.py \
  --src-dir data/processed/diarios_txt \
  --out-dir data/processed/diarios_txt_balanced \
  --keep-fraction 0.5 --seed 42
```

| Conjunto | Docs | Licitação (docs) | Licitação (tokens) | Tokens totais (palavras) |
|----------|------|------------------|--------------------|--------------------------|
| original (`diarios_txt`) | 2000 | 731 (36.5%) | 42.5% | 2.95M |
| balanceado (`diarios_txt_balanced`) | 1635 | 366 (22.4%) | 27.0% | 2.32M (79% do original) |

Observação: por contagem de documentos a licitação já era minoria (36.5%), mas por
tokens chegava a 42.5% (documentos de licitação são mais longos: 1715 vs 1338
palavras em média). O downsample de 50% derruba a participação em tokens de 42.5%
para 27.0%.

## Held-out balanceado

Mesmo pool disjunto do treino (os 150 docs de `diarios_heldout.jsonl`, gerados com
`diarios_to_text.py --skip 2000`), com o mesmo downsample de licitação aplicado
(`downsample_licitacao(..., keep_fraction=0.5, seed=42)`): resultado em
`data/processed/diarios_heldout_balanced.jsonl`, 125 docs (24 licitação, 101
outros). Continua disjunto do treino (vem de depois do skip 2000).

## Protocolo de avaliação

Para isolar o efeito da limpeza, o mesmo modelo (Qwen3-0.6B-Base) é treinado no
corpus completo e no balanceado, com recipe idêntica (só muda o `input_dir` e o
`output_dir`; ver `configs/pretrain_diarios_qwen3_0p6b_balanced.yaml`). Avalia-se
cruzando os dois held-outs (original e balanceado), perplexidade menor melhor.

Confound a registrar: o corpus balanceado tem ~21% menos tokens; parte de qualquer
diferença pode vir de ver menos dados, não só de remover licitação. Por isso a
leitura considera os dois held-outs e não só o número absoluto.

## Resultado

Qwen3-0.6B-Base treinado nos dois corpora, perplexidade (menor melhor):

| Corpus de treino | held-out original | held-out balanceado | QA |
|------------------|-------------------|---------------------|-----|
| (base, antes) | 11.47 | 11.45 | 11.58 |
| completo | 6.88 | 6.86 | 10.13 |
| balanceado | 7.16 | 7.09 | 10.29 |

Podar licitação **não ajudou; piorou** em todos os conjuntos, inclusive no held-out
balanceado (6.86 do completo vs 7.09 do balanceado), onde o modelo balanceado
estaria na própria distribuição. Pesa contra a poda: ~21% menos tokens de treino e o
fato de atos de licitação serem formulaicos (fáceis de prever, ajudam a
perplexidade). É o oposto do RAG, onde a repetição das licitações prejudicava a
diversidade da recuperação. A Q1 fica com o corpus completo; o balanceado serve só
como ablação de diagnóstico. O corpus original não foi alterado.

Resultados consolidados em `results/q1_balanceamento_licitacao.csv` e
`results/README.md`.
