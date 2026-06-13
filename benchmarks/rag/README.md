# Benchmark de RAG (Q5 - GraphRAG agentico sobre os diarios)

Conjunto de 30 perguntas para avaliar a contribuicao do RAG (Questao 5) sobre o
corpus de diarios oficiais dos municipios do Piaui. Construido a partir do indice
GraphRAG ja montado, entao as respostas de referencia sao ancoradas no conteudo
indexado.

## Arquivo

- `diarios_rag_30.jsonl` - um objeto por linha: `{"question", "expected_answer",
  "type"}`. `type` e `factual` (resposta em um unico trecho) ou `multihop`
  (precisa combinar duas relacoes do grafo). Gerado por
  `scripts/make_rag_benchmark.py` (git-ignored se regenerado; versionar a versao
  curada).

## Por que perguntas especificas do dominio

As perguntas pedem fatos concretos dos diarios (numeros de portaria, CNPJ, valores,
nomes nomeados, datas). Um LLM sem recuperacao nao tem como saber esses fatos
(nao estao no treino dele), entao o baseline sem RAG tende a 0; o RAG so acerta se
recuperar o trecho/relacao certa. Isso isola a contribuicao do RAG.

## Cenarios avaliados (`scripts/eval_rag.py --modes ...`)

- `baseline` (sempre): o LLM responde sem acesso ao indice.
- `standard`: RAG comum, sem agente e sem grafo (recupera por vetor uma vez e gera).
- `agentic_vector`: agente self-reflexivo (LangGraph), so com recuperacao vetorial.
- `agentic_graph`: GraphRAG completo (agente self-reflexivo + grafo de conhecimento).

Comparar os quatro isola cada peca: recuperacao (standard vs baseline), loop de
agente (agentic_vector vs standard) e grafo (agentic_graph vs agentic_vector). Cada
modo e uma classe `RagRunner` em `src/llm_finetuning/rag/pipelines.py` (Strategy/OCP:
um modo novo = uma classe nova registrada, sem mudar o avaliador).

Cada resposta e pontuada de 0 a 5 por um LLM-as-judge contra a `expected_answer`.
A analise reporta o acerto medio de cada cenario, o ganho sobre o baseline e a taxa
de auto-correcao do no Self-Reflective. Saida em `results/benchmark_rag_*.csv`.

## Como rodar

```bash
# 1. Construir o indice (vetor + grafo) sobre um subconjunto dos diarios
CUDA_VISIBLE_DEVICES=0 python scripts/build_rag_index.py --config configs/rag_diarios.yaml
# 2. (Opcional) regenerar o benchmark ancorado no indice
CUDA_VISIBLE_DEVICES=0 python scripts/make_rag_benchmark.py --config configs/rag_diarios.yaml
# 3. Avaliar (comparar os 3 modos de RAG contra o baseline)
CUDA_VISIBLE_DEVICES=0 python scripts/eval_rag.py --config configs/rag_diarios.yaml \
  --modes standard,agentic_vector,agentic_graph --out results/benchmark_rag_compare_qwen8b.csv
```

Trocar de motor/familia e so apontar para outro config (ex.:
`configs/rag_diarios_gemma3_1b_it.yaml`), reusando o mesmo indice.

## Motor (LLM) e limitacoes

O motor padrao e o `Qwen3-8B` instruct em bf16 numa unica L4 (rapido). A variante
`configs/rag_diarios_qwen3_30b.yaml` usa o `Qwen3-30B-A3B-Instruct-2507-FP8` nas
duas L4, reservada para quando o multi-GPU (NCCL) for corrigido, pois hoje so roda
em model-parallel ingenuo (~28 s/geracao). O benchmark e gerado automaticamente:
algumas respostas de referencia podem ser ruidosas; uma curadoria manual e um passo
de melhoria futuro.
