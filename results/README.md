# Resultados consolidados (ledger geral)

Registro acumulado de todas as execuções de treino/avaliação das 6 questões, para o
relatório e para comparar modelos, tamanhos e métodos sem perder corridas anteriores.
Nenhum resultado é descartado. Cada questão tem sua seção abaixo e um CSV próprio em
`results/`; os números intrínsecos de cada corrida ficam em
`benchmarks/<fase>/results/<antes|depois>/*.json` (git-ignored).

## Visão geral (Q1-Q6)

| Q | Tema | Status | Resultado principal | CSV |
|---|------|--------|---------------------|-----|
| Q1 | Pré-treino contínuo (full-param) | feito (0.6B, 1.7B, gemma; 4B na fila) | base fine-tunado >> instruct; gemma-pt depois 5.49 (melhor da escada); podar licitação **piora** | `runs.csv`, `q1_base_vs_instruct.csv`, `q1_balanceamento_licitacao.csv` |
| Q2 | Pós-treino SFT | feito (0.6B, 1.7B, gemma; 4B na fila) | SFT baixa a ppl em todos; gemma 0.67->1.57 no juiz; **Q1+SFT > SFT** no Qwen | `q2_sft.csv` |
| Q3 | LoRA (PEFT) | feito | **LoRA iguala/supera o SFT pleno** treinando ~1.7% dos params | `q3_lora.csv` |
| Q4 | Destilação teacher->student | feito | transferência: SmolLM2-135M 0.07->0.34, gemma 84% do gap; **logit-KD ~ response-based** | `q4_distill.csv`, `q4_methods.csv` |
| Q5 | RAG (3 modos x motores) | feito (30B na fila) | baseline ~1.1 -> RAG ~2.7; a recuperação é o ganho; exemplos qualitativos | `benchmark_rag_*.csv`, `q5_qualitativos.md` |
| Q6 | Guardrails | feito | proteção **0->100%** das adversariais/PII, **0 falsos positivos** nas benignas | `q6_guardrails.csv` |

Pendente (preso na fila do gpunode01, 2 GPUs): 4B da Q1 e Q2 e o motor 30B da Q5.
Detalhes e leituras por questão abaixo.

## Variação dos resultados (amplitude entre modelos/métodos)

O projeto testou muitos modelos e abordagens; a dispersão dos resultados é parte da
evidência. Amplitudes observadas (detalhe nas seções):

- **Q1 (held-out ppl depois, menor melhor):** de **5.49** (gemma-3-1b-pt) a **6.88**
  (Qwen3-0.6B); base antes 8.59-11.47. Base vs instruct sem treino: 0.6B 6.88 (base
  FT) vs 16.30 (instruct); gemma 5.49 vs 28.21. Corpus cheio vs licitação-podado:
  6.88 vs 7.16 (podar piora).
- **Q2 (juiz 0-5 no recall):** SFT de **1.49** (0.6B) a **1.89** (1.7B); o ganho do
  SFT sobre o base varia de **+0.9** (gemma, base fraco 0.67->1.57) a **~0** (Qwen,
  base ja forte). ppl da resposta cai em todos (ex.: 1.7B 7.44->5.09).
- **Q3 (LoRA vs SFT pleno, juiz):** LoRA vence ou empata em **5 de 6** casos; delta
  de **+0.20** (0.6B base) a **-0.01**; ~1.7% dos params treinados.
- **Q4 (transfer ratio):** de **0.33** (SmolLM2-360M) a **0.84** (gemma); juiz dos
  students de 0.07 (135M base) a 0.66 (teacher 8B); ppl da resposta cai de ~5% a ~58%
  conforme o student. Métodos response-based vs logit-KD: empate (0.51 vs 0.50).
- **Q5 (juiz 0-5, modo standard):** por motor, de **0.73** (gemma-3-1b-pt base) a
  **2.70** (Qwen3-8B); por modo no 8B, 2.60-2.70 (recuperação satura). Baseline sem
  RAG ~1.1.
- **Q6 (taxa com guardrails):** 100% nas 3 categorias adversariais (jailbreak,
  inseguro, PII) e 0% de falso positivo nas benignas; sem guardrails, 0% de proteção.

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
instruct de prateleira sem treino. Os instruct Qwen na versão não-base
(`Qwen3-0.6B/1.7B/4B/8B`) e o `gemma-3-1b-it` foram só avaliados, sem fine-tuning;
os `-base`/`-pt` têm antes e depois. Tabela completa em
`results/q1_base_vs_instruct.csv`. Held-out, perplexidade (menor melhor):

| Tam. (família) | base antes | base depois (FT) | instruct sem FT |
|----------------|------------|------------------|-----------------|
| 0.6B (qwen3) | 11.47 | **6.88** | 16.30 |
| 1.7B (qwen3) | 8.59 | **5.73** | 11.92 |
| 4B (qwen3) | 7.17 | (job SLURM 399) | 10.02 |
| 8B (qwen3) | - | - | 8.17 |
| 1.0B (gemma3) | 9.57 | **5.49** | 28.21 |

(qwen3 0.6B/1.7B/4B: par base/instruct do mesmo tamanho; 8B só instruct. gemma3:
`-pt` base e `-it` instruct, par da mesma família 1B.)

Leituras:
- **O base fine-tunado vence o instruct do mesmo tamanho com folga**: 0.6B 6.88 vs
  16.30; 1.7B 5.73 vs 11.92. O pré-treino contínuo no domínio supera o pós-treino
  de chat para esta tarefa intrínseca.
- **Tamanho não compensa domínio**: o `Qwen3-1.7B-Base` fine-tunado (5.73) e até o
  `Qwen3-0.6B-Base` fine-tunado (6.88) batem o `Qwen3-8B` instruct sem treino
  (8.17), um modelo 5x a 13x maior.
- **Base < instruct já no ponto de partida**: em todo tamanho, o base antes tem
  perplexidade menor que o instruct sem treino (0.6B 11.47 < 16.30; 1.7B 8.59 <
  11.92; 4B 7.17 < 10.02). O `Qwen3-4B-Base` cru (7.17) já supera o `Qwen3-8B`
  instruct (8.17). O alinhamento de chat cobra um imposto em texto cru de diário,
  monotônico nas duas famílias (gemma-it no extremo, 28.21).
- Confirma quantitativamente, em duas famílias, a decisão de partir de modelos
  **base** nas Q1-Q3. O `Qwen3-4B-Base` depois entra aqui quando o job 399 fechar.

### Mini análise (Q1)

Três efeitos se somam e apontam na mesma direção:

1. **Adaptação de domínio supera escala.** Para perplexidade em texto de diário, o
   que mais importa não é o tamanho do modelo, é ter visto o domínio. Um base
   pequeno fine-tunado (Qwen3-1.7B-Base 5.73; gemma-3-1b-pt 5.49) bate um instruct
   várias vezes maior sem treino (Qwen3-8B 8.17). Em um orçamento de 2x L4, treinar
   um base pequeno rende mais que pegar um instruct grande de prateleira.
2. **O ponto de partida já favorece o base.** Antes de qualquer treino, o base tem
   perplexidade menor que o instruct do mesmo tamanho em todos os pontos medidos; o
   pós-treino de chat afasta o modelo de texto cru (imposto de alinhamento),
   monotônico nas duas famílias. A magnitude depende da família: no Qwen o instruct
   0.6B sobe para 16.30 (base antes 11.47), no gemma o `-it` dispara para 28.21
   (base `-pt` 9.57).
3. **A escolha base vs instruct pesa mais que a família.** O melhor (gemma-pt
   treinado, 5.49) e o pior (gemma-it cru, 28.21) são a mesma família 1B: a decisão
   de partir do base muda o resultado mais que trocar de arquitetura.

Implicação para o projeto: a decisão de usar modelos `-base`/`-pt` nas Q1-Q3 está
validada por evidência em duas famílias, e a escada de tamanho ainda paga (cada
base maior parte e termina mais baixo). Ressalva: a perplexidade premia o estilo
formulaico do diário; por isso a leitura principal é a perplexidade, com a acurácia
de token como apoio.

### Ablação: podar licitações do corpus de treino ajuda a Q1?

Hipótese (vinda do estudo de licitações no RAG): as licitações, repetitivas,
poderiam estar "poluindo" o treino. Construiu-se um corpus balanceado mexendo
**só** nas licitações (todos os não-licitação mantidos, 50% das licitações
sorteadas, seed 42; método em `docs/DATASET_BALANCEAMENTO.md`): 2000 -> 1635 docs,
licitação de 42.5% para 27.0% dos tokens. Treinou-se o mesmo Qwen3-0.6B-Base nos
dois corpora (recipe idêntica) e avaliou-se cruzando dois held-outs disjuntos: o
original (distribuição cheia) e um balanceado (licitação reduzida). Perplexidade,
menor melhor; dados em `results/q1_balanceamento_licitacao.csv`.

| Corpus de treino | held-out original | held-out balanceado | QA |
|------------------|-------------------|---------------------|-----|
| (base, antes) | 11.47 | 11.45 | 11.58 |
| completo | **6.88** | **6.86** | **10.13** |
| balanceado | 7.16 | 7.09 | 10.29 |

Conclusão: **podar licitação não ajuda a Q1; piora**. O modelo treinado no corpus
completo vence em todos os conjuntos, inclusive no held-out balanceado (6.86 vs
7.09), onde o modelo balanceado teria a melhor chance por estar na própria
distribuição. Dois efeitos somam contra a poda: (1) o corpus balanceado tem ~21%
menos tokens (ver menos texto piora a perplexidade); (2) atos de licitação são
formulaicos e previsíveis, então até ajudam o modelo de linguagem em vez de
atrapalhar. É o oposto do RAG: lá a repetição das licitações prejudicava a
diversidade da recuperação; aqui, para prever o próximo token, mais texto do
domínio (mesmo repetitivo) ajuda. Recomendação: a Q1 fica com o corpus completo; o
balanceado é mantido só como ablação de diagnóstico (não se apaga o original).

## Q2 - pós-treino (SFT) sobre o docentesDC

>= 1.000 pares `{instruction, input?, output}` gerados do dataset oficial
`vickminari/docentesDC` (Qwen3-8B como gerador), SFT full-parameter com loss só na
resposta. Avaliação antes/depois num held-out de **recall in-domain** (perguntas
novas sobre os mesmos textos-fonte do treino, sem as perguntas de treino): juiz fixo
Qwen3-8B (0-5) e perplexidade da resposta (menor melhor). Dados em
`results/q2_sft.csv`. Experimento A/B: SFT partindo do **base** vs do **checkpoint
da Q1** (pré-treino contínuo), para ver se Q1 e Q2 se somam.

| Modelo | juiz base | juiz SFT | juiz SFT(Q1) | ppl base | ppl SFT |
|--------|-----------|----------|--------------|----------|---------|
| Qwen3-0.6B | 1.49 | 1.49 | 1.61 | 9.29 | **6.44** |
| Qwen3-1.7B | 1.88 | 1.89 | **1.99** | 7.44 | **5.09** |
| gemma-3-1b | **0.67** | **1.57** | 1.47 | 10.95 | **7.38** |

Leituras:
- **O SFT baixa a perplexidade da resposta em todos** (9.29->6.44, 7.44->5.09,
  10.95->7.38): o modelo aprende a distribuição das respostas do domínio docente.
  Esse é o antes/depois mais limpo.
- **O ganho no juiz depende do base.** O `gemma-3-1b-pt` é fraco em seguir instrução
  (0.67/5) e o SFT dá um salto grande (->1.57, +133%): demonstração clara de que o
  SFT funciona. Os Qwen base já respondem (1.5-1.9), então o ganho marginal do SFT
  no juiz é pequeno (o base já faz a maior parte). A perplexidade, teacher-forced,
  separa mesmo quando a geração greedy de um modelo pequeno ainda erra os fatos
  específicos.
- **Q1 + SFT vs SFT puro (juiz):** ajuda no Qwen (sft_q1 > sft_base em 0.6B 1.61 vs
  1.49 e em 1.7B 1.99 vs 1.89), neutro/levemente negativo no gemma. Lean positivo de
  que o pré-treino contínuo (Q1) e o SFT (Q2) se somam.
- **Escala ajuda** (1.7B > 0.6B em juiz e ppl). O 4B (FSDP+offload, SLURM 2-GPU)
  fica como o tamanho maior a rodar.
- Caveat de design: um held-out de conteúdo disjunto deixa o juiz chato (modelo não
  pode saber fatos inéditos); por isso usa-se o recall in-domain. As duas leituras
  estão no CSV (`eval_set` = recall vs disjoint).

## Q3 - pós-treino LoRA (PEFT) vs SFT pleno

Mesmo dataset, held-out de recall e juiz da Q2; só muda o método (LoRA r=16,
~1.7% dos params, mesclado no fim). Mesma escada (0.6B, 1.7B, gemma) e A/B (base vs
checkpoint Q1). Juiz 0-5 / perplexidade da resposta (menor melhor). Dados em
`results/q3_lora.csv`.

| Modelo (start) | juiz SFT pleno | juiz LoRA | ppl SFT pleno | ppl LoRA |
|----------------|----------------|-----------|---------------|----------|
| 0.6B (base) | 1.49 | **1.69** | 6.44 | 6.29 |
| 0.6B (Q1) | 1.61 | 1.60 | 6.76 | 6.39 |
| 1.7B (base) | 1.89 | **2.05** | 5.09 | 5.08 |
| 1.7B (Q1) | 1.99 | **2.11** | 5.14 | 5.09 |
| gemma (base) | 1.57 | **1.67** | 7.38 | 7.52 |
| gemma (Q1) | 1.47 | **1.65** | 7.55 | 7.68 |

Leituras:
- **LoRA iguala ou supera o SFT pleno** no juiz em 5 de 6 casos (empata 1),
  treinando ~1.7% dos parâmetros. Provável regularização: com só 1.000 exemplos, o
  full fine-tune de um modelo pequeno tende a overfit/forget, e o LoRA segura isso.
- Perplexidade comparável entre os dois (LoRA levemente melhor no Qwen, levemente
  pior no gemma).
- Conclusão da Q3: **PEFT alcança a qualidade do fine-tune pleno a uma fração do
  custo** (params, memória, tempo), confirmando o valor de LoRA/QLoRA. QLoRA e o 4B
  (via SLURM) ficam como extensões (modelos maiores em 1 GPU).

## Q4 - destilação de conhecimento (teacher -> student)

Teacher = `Qwen3-8B` (gera Q&A sintético ancorado nos diários, estilo RAG); students
= escada de modelos menores de boas fontes (`SmolLM2-135M/360M`, `Qwen2.5-0.5B`,
`Qwen3-0.6B-Base`, `gemma-3-1b-pt`), destilados por response-based SFT nos dados do
teacher. Benchmark de 100 perguntas (recall in-domain dos diários), juiz 0-5 e
perplexidade da resposta. **Transfer ratio** = (distill - base)/(teacher - base) =
fração do gap teacher-student fechado. Dados em `results/q4_distill.csv`.

| Student | params | base juiz | distill juiz | transfer | base ppl | distill ppl |
|---------|--------|-----------|--------------|----------|----------|-------------|
| SmolLM2-135M | 135M | 0.07 | **0.34** | 0.46 | 22.2 | 19.9 |
| SmolLM2-360M | 360M | 0.18 | **0.34** | 0.33 | 12.6 | 11.7 |
| Qwen2.5-0.5B | 0.5B | 0.34 | **0.46** | 0.38 | 12.4 | **6.3** |
| Qwen3-0.6B | 0.6B | 0.60 | 0.51 | base~teacher | 10.6 | **6.1** |
| gemma-3-1b | 1.0B | 0.41 | **0.62** | **0.84** | 11.3 | **4.6** |

(teacher `Qwen3-8B`: juiz 0.66, ppl 10.6.)

Leituras:
- **Houve transferência de conhecimento:** 4 dos 5 students sobem no juiz. O 135M vai
  de ~0 (0.07) para 0.34, fechando 46% do gap teacher-student; o gemma fecha **84%**
  do gap (0.41 -> 0.62, quase alcançando o teacher 0.66).
- **A perplexidade da resposta despenca em todos** (135M 22->20; qwen2.5 12->6; gemma
  11->4.6): os students absorveram a distribuição de respostas do teacher, o sinal
  central da destilação.
- **Lei de escala da destilação:** students mais fracos/menores têm mais espaço e
  mostram transferência clara; o `Qwen3-0.6B` base já estava ~ teacher (0.60 vs 0.66),
  sem gap a fechar, então a destilação não moveu o juiz (mas baixou muito a ppl).
- **Ressalva:** o teacher (8B closed-book) marca só 0.66 neste recall difícil, um teto
  fraco. Um teacher com RAG (recuperando os diários) elevaria o teto e a margem de
  transferência (diferencial B).

### Response-based vs logit-KD (Qwen3-0.6B)

Comparação dos dois paradigmas de destilação no mesmo student (`results/q4_methods.csv`):

| Método | teacher | juiz | ppl resposta |
|--------|---------|------|--------------|
| base (sem destilar) | - | 0.60 | 10.56 |
| response-based (SFT nas respostas) | Qwen3-8B | 0.51 | 6.12 |
| logit-KD (KL nos logits) | Qwen3-1.7B | 0.50 | 6.51 |
| teacher (referência) | - | 0.66 | 10.56 |

Leituras:
- **Os dois métodos empatam** no student 0.6B (juiz 0.51 vs 0.50; ppl 6.12 vs 6.51),
  mesmo o logit-KD usando um teacher menor (1.7B vs 8B). Ambos derrubam fortemente a
  perplexidade da resposta (10.56 -> ~6.3), transferindo a distribuição do teacher.
- Nenhum move o juiz para cima aqui porque o `Qwen3-0.6B` base já estava ~ teacher
  (0.60 vs 0.66) neste recall: sem gap a fechar. O ganho da destilação aparece na
  perplexidade, e (na tabela anterior) no juiz dos students mais fracos.

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

### Exemplos qualitativos (sem RAG vs com RAG)

`results/q5_qualitativos.md` (gerado por `scripts/eval_rag_examples.py`) mostra
respostas lado a lado. Padrão claro: **sem RAG o modelo alucina** (inventa "artigo 42
do CPC" para uma pergunta sobre os diários) ou **se recusa** ("preciso de mais
informações"); **com RAG ele ancora** na evidência recuperada e acerta ou chega perto
(ex.: número de lei, datas, itens normativos). Ilustra a contribuição do RAG e também
seus limites (algumas respostas ainda imprecisas, em parte por perguntas
auto-geradas ambíguas, ex.: "qual a data de assinatura do documento?" sem dizer qual).

## Q6 - guardrails (camada de proteção)

Camada `guardrails/` (filtros componíveis registrados, OCP): bloqueio de jailbreak e
de pedidos inseguros na entrada, e mascaramento de PII brasileiro (CPF, CNPJ, CEP,
telefone, email) na saída. Benchmark de 30 perguntas
(`benchmarks/guardrails/guardrails_30.jsonl`): 10 adversariais (jailbreak/inseguro),
5 com PII na saída, 15 benignas. Avaliação com vs sem a camada
(`scripts/eval_guardrails.py`, sem LLM, isola a proteção). Dados em
`results/q6_guardrails.csv`.

| Tipo | n | sem guardrails | com guardrails |
|------|---|----------------|----------------|
| jailbreak (bloquear) | 5 | 0/5 | **5/5** |
| inseguro (bloquear) | 5 | 0/5 | **5/5** |
| PII na saída (mascarar) | 5 | 0/5 | **5/5** |
| benigna (passar) | 15 | 15/15 | 15/15 |

Leituras:
- **Grau de proteção:** de 0% para **100%** de bloqueio/mascaramento das entradas
  adversariais e da PII, **sem nenhum falso positivo** nas 15 benignas (helpfulness
  preservado). A camada resolve o dilema helpfulness vs harmlessness neste conjunto.
- **Ressalva honesta (robustez adversarial medida):** os filtros são heurísticos
  (regex/marcadores), então pegam padrões conhecidos. Num teste com 10 ataques
  **parafraseados** (mesma intenção, redação fora da blocklist;
  `benchmarks/guardrails/guardrails_adversarial.jsonl`,
  `results/q6_adversarial.csv`), a taxa de bloqueio cai de 100% para **0%**: a camada
  por regra é frágil a reformulações. Um guardrail classificador por modelo (entra
  como outro filtro registrado, sem mudar a camada) generalizaria melhor; fica como
  extensão. O mascaramento de PII por regex é robusto para os formatos brasileiros
  padronizados.

## Convenção de colunas (runs.csv)

`date, question, model, params, variant (base|instruct|vlm), modality (text|vlm),
method, dataset, eval_set, ppl_before, ppl_after, ce_before, ce_after,
tokacc_before, tokacc_after, train_loss, blocks, gpus, config, notes`.
