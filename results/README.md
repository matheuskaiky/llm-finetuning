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

Held-out (texto de diário inédito, disjunto do treino), perplexidade (menor melhor):

| Modelo | Tam. | Variante | PPL antes | PPL depois | TokAcc depois |
|--------|------|----------|-----------|------------|---------------|
| Qwen3-0.6B-Base | 0.6B | base | 11.47 | 6.88 | 0.603 |
| Qwen3-1.7B-Base | 1.7B | base | 8.59 | **5.73** | 0.627 |
| gemma-3-1b-pt | 1.0B | base | 9.57 | **5.49** | 0.637 |
| gemma-3-1b-it | 1.0B | instruct | **28.21** | 6.87 | 0.609 |

(qa conceitual e métricas completas em `runs.csv`.)

Leituras:
- **Escada de tamanho**: o modelo maior parte de base melhor e chega mais baixo
  depois (Qwen3 0.6B 6.88 -> 1.7B 5.73). O 4B está pronto mas depende do multi-GPU
  (ver `docs/SUPORTE_INFRA_MULTIGPU.md`).
- **Cross-family**: o `gemma-3-1b-pt` (1B) termina melhor que o Qwen3-1.7B (5.49 vs
  5.73) apesar de menor; arquitetura/tokenizer da família importam.
- **Base vs instruct (evidencia forte)**: o `gemma-3-1b-it` (instruct) começa MUITO
  pior em texto de diário (perplexidade 28.2 vs 9.6 do base irmao) porque o
  pos-treino de chat o afasta de texto cru, e mesmo após o pré-treino contínuo
  termina pior que o base (6.87 vs 5.49). Confirma empiricamente a escolha de partir
  de modelos **base** nas Q1-Q3.
- A acurácia de token é otimista no texto formulaico; a perplexidade é mais
  informativa.

### Base fine-tunado vs instruct sem fine-tuning

Comparação direta da escolha da equipe (partir de modelos base) contra usar um
instruct de prateleira sem treino. Os instruct Qwen na versão nao-base
(`Qwen3-0.6B/1.7B/4B/8B`) e o `gemma-3-1b-it` foram só avaliados, sem fine-tuning;
os `-base`/`-pt` têm antes e depois. Tabela completa em
`results/q1_base_vs_instruct.csv`. Held-out, perplexidade (menor melhor):

| Tam. (familia) | base antes | base depois (FT) | instruct sem FT |
|----------------|------------|------------------|-----------------|
| 0.6B (qwen3) | 11.47 | **6.88** | 16.30 |
| 1.7B (qwen3) | 8.59 | **5.73** | 11.92 |
| 4B (qwen3) | 7.17 | (job SLURM 399) | 10.02 |
| 8B (qwen3) | - | - | 8.17 |
| 1.0B (gemma3) | 9.57 | **5.49** | 28.21 |

(qwen3 0.6B/1.7B/4B: par base/instruct do mesmo tamanho; 8B so instruct. gemma3:
`-pt` base e `-it` instruct, par da mesma familia 1B.)

Leituras:
- **O base fine-tunado vence o instruct do mesmo tamanho com folga**: 0.6B 6.88 vs
  16.30; 1.7B 5.73 vs 11.92. O pré-treino contínuo no dominio supera o pos-treino
  de chat para esta tarefa intrinseca.
- **Tamanho nao compensa dominio**: o `Qwen3-1.7B-Base` fine-tunado (5.73) e ate o
  `Qwen3-0.6B-Base` fine-tunado (6.88) batem o `Qwen3-8B` instruct sem treino
  (8.17), um modelo 5x a 13x maior.
- **Base < instruct ja no ponto de partida**: em todo tamanho, o base antes tem
  perplexidade menor que o instruct sem treino (0.6B 11.47 < 16.30; 1.7B 8.59 <
  11.92; 4B 7.17 < 10.02). O `Qwen3-4B-Base` cru (7.17) ja supera o `Qwen3-8B`
  instruct (8.17). O alinhamento de chat cobra um imposto em texto cru de diario,
  monotonico nas duas familias (gemma-it no extremo, 28.21).
- Confirma quantitativamente, em duas familias, a decisao de partir de modelos
  **base** nas Q1-Q3. O `Qwen3-4B-Base` depois entra aqui quando o job 399 fechar.

### Mini analise (Q1)

Tres efeitos se somam e apontam na mesma direcao:

1. **Adaptacao de dominio supera escala.** Para perplexidade em texto de diario, o
   que mais importa nao e o tamanho do modelo, e ter visto o dominio. Um base
   pequeno fine-tunado (Qwen3-1.7B-Base 5.73; gemma-3-1b-pt 5.49) bate um instruct
   varias vezes maior sem treino (Qwen3-8B 8.17). Em um orcamento de 2x L4, treinar
   um base pequeno rende mais que pegar um instruct grande de prateleira.
2. **O ponto de partida ja favorece o base.** Antes de qualquer treino, o base tem
   perplexidade menor que o instruct do mesmo tamanho em todos os pontos medidos; o
   pos-treino de chat afasta o modelo de texto cru (imposto de alinhamento),
   monotonico nas duas familias. A magnitude depende da familia: no Qwen o instruct
   0.6B sobe para 16.30 (base antes 11.47), no gemma o `-it` dispara para 28.21
   (base `-pt` 9.57).
3. **A escolha base vs instruct pesa mais que a familia.** O melhor (gemma-pt
   treinado, 5.49) e o pior (gemma-it cru, 28.21) sao a mesma familia 1B: a decisao
   de partir do base muda o resultado mais que trocar de arquitetura.

Implicacao para o projeto: a decisao de usar modelos `-base`/`-pt` nas Q1-Q3 esta
validada por evidencia em duas familias, e a escada de tamanho ainda paga (cada
base maior parte e termina mais baixo). Ressalva: a perplexidade premia o estilo
formulaico do diario; por isso a leitura principal e a perplexidade, com a acuracia
de token como apoio.

## Q5 - RAG (ablação de 3 modos x 3 motores)

Benchmark de 30 perguntas (`benchmarks/rag/diarios_rag_30.jsonl`), pontuadas 0-5 por
um **juiz fixo Qwen3-8B** (igual para todos os motores, para comparação justa).
Modos: baseline (sem RAG), standard (recupera+gera), agentic_vector (agente, sem
grafo), agentic_graph (agente + grafo). CSVs em `results/benchmark_rag_compare_*.csv`.

Acerto médio (0-5), corpus cheio, **roteador corrigido** (grafo sempre consultado em
modo grafo) e benchmark multi-hop sem vazamento:

| Motor | baseline | standard | agentic_vector | agentic_graph |
|-------|----------|----------|----------------|---------------|
| Qwen3-8B (instruct) | 1.10 | **2.70** | 2.60 | 2.63 |
| gemma-3-1b-it (instruct) | 0.67 | 2.07 | **2.23** | 2.03 |
| gemma-3-1b-pt (base) | 0.47 | 0.73 | 0.73 | **0.87** |

Leituras:
- **RAG ajuda todo motor**: o maior salto é o `standard` sobre o baseline; a
  recuperação é o ganho principal.
- **Grafo/agente quase não separam do standard**, mesmo após corrigir o roteador
  (que antes deixava o grafo de fora em 29/30). No 8B a recuperação simples satura
  (2.70) e o agente/grafo não melhoram. Conclusão honesta: nesta tarefa (achar um
  fato em texto, respondido por um motor forte) o grafo/multi-hop agrega pouco; não
  era só bug do roteador.
- **Motor importa, instruct > base para RAG**: 8B > gemma-it > gemma-pt.
- **Juiz fixo foi essencial**: com auto-julgamento o gemma-it inflava para baseline
  2.87 e o RAG aparecia negativo (artefato do juiz 1B, não do método).
- Ablação do dataset balanceado (licitações podadas) em `*_balanced_*` testa se a
  repetição das licitações achatava a diferença entre os modos.

### Ablação: corpus cheio vs balanceado (licitações podadas)

Dataset balanceado = 75 licitação / 75 outros (grafo mais rico: 411 entidades vs
202). Mesma metodologia (3 modos, juiz fixo 8B), benchmark próprio ancorado.

| Motor | modo | cheio | balanceado |
|-------|------|-------|------------|
| Qwen3-8B | baseline | 1.10 | 1.77 |
| Qwen3-8B | standard | 2.70 | 3.50 |
| Qwen3-8B | agentic_graph | 2.63 | 3.37 |
| gemma-1b-it | standard | 2.07 | 2.40 |
| gemma-1b-it | agentic_graph | 2.03 | 2.70 |
| gemma-1b-pt | standard | 0.73 | 0.87 |

Conclusões: podar as licitações repetitivas **eleva a recuperação** de forma clara
(8B standard 2.70 -> 3.50), confirmando que elas poluíam o índice. Mas isso ajuda o
**retrieval**, não especificamente o grafo (os modos seguem próximos). Em produção
não se descartam licitações: estratégias para preservá-las em `docs/RAG_ROADMAP.md`.
O `gemma-1b-pt` (base) chega a piorar com RAG (standard 0.87 < baseline 1.10): um
modelo base fraco se confunde com o contexto recuperado.

## Convenção de colunas (runs.csv)

`date, question, model, params, variant (base|instruct|vlm), modality (text|vlm),
method, dataset, eval_set, ppl_before, ppl_after, ce_before, ce_after,
tokacc_before, tokacc_after, train_loss, blocks, gpus, config, notes`.
