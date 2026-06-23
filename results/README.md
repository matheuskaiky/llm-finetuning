# Resultados consolidados (ledger geral)

Registro acumulado de todas as execuções de treino/avaliação das 6 questões, para o
relatório e para comparar modelos, tamanhos e métodos sem perder corridas anteriores.
Nenhum resultado é descartado. Cada questão tem sua seção abaixo e um CSV próprio em
`results/`; os números intrínsecos de cada corrida ficam em
`benchmarks/<fase>/results/<antes|depois>/*.json` (git-ignored).

Os gráficos e a análise de dados consolidada ficam em
`notebooks/graficos_resultados.ipynb` (lê estes CSVs): leitura por questão mais a
síntese transversal (leaderboard SFT vs LoRA, distribuição das notas por motor no RAG,
vitória/empate/derrota do RAG por pergunta, desempenho por tipo, mapa família x tarefa,
escala x qualidade). O notebook é regenerado por `scripts/_build_results_notebook.py`.

Cobertura das métricas exigidas: a Q1 tem as três métricas (perplexidade, entropia
cruzada e acurácia de token) antes/depois tanto no held-out de texto de diário quanto no
benchmark de P&R (perguntas e respostas conceituais, base vs instruct sem treino); Q2/Q3
e Q4 também mostram a perplexidade da resposta (antes/depois), além do juiz.

Ressalva de método: os gabaritos dos benchmarks atuais foram gerados com IA (o juiz
também é um LLM, Qwen3-8B 0-5). Falta um conjunto de referência feito a mão, sem IA,
como gabarito independente (tarefa em aberto). Os números abaixo devem ser lidos com
essa limitação em mente.

## Visão geral (Q1-Q6)

| Q | Tema | Status | Resultado principal | CSV |
|---|------|--------|---------------------|-----|
| Q1 | Pré-treino contínuo (full-param) | feito (0.6B, 1.7B, gemma) | base fine-tunado >> instruct; gemma-pt depois 5.49 (melhor da escada); podar licitação **piora** | `runs.csv`, `q1_base_vs_instruct.csv`, `q1_balanceamento_licitacao.csv`, `q1_forgetting.csv` |
| Q2 | Pós-treino SFT | feito (0.6B, 1.7B, gemma) | SFT baixa a ppl em todos; gemma 0.67->1.57 no juiz; **Q1+SFT > SFT** no Qwen | `q2_sft.csv` |
| Q3 | LoRA (PEFT) | feito | **LoRA iguala/supera o SFT pleno** treinando ~1.7% dos params | `q3_lora.csv` |
| Q4 | Destilação teacher->student | feito | transferência: SmolLM2-135M 0.07->0.34, gemma 84% do gap; **logit-KD ~ response-based** | `q4_distill.csv`, `q4_methods.csv` |
| Q5 | RAG (3 modos x motores) | feito (inclui motor 30B) | baseline ~1.1 -> RAG ~2.7; a recuperação é o ganho; exemplos qualitativos; retrieval hit-rate | `benchmark_rag_*.csv`, `q5_rag_30b.csv`, `q5_retrieval.csv`, `q5_qualitativos.md` |
| Q6 | Guardrails | feito | proteção **0->100%** das adversariais/PII, **0 falsos positivos** nas benignas | `q6_guardrails.csv` |

Limite de hardware: o 4B em full fine-tuning (Q1/Q2) **não cabe nas 2x L4** de 22 GB
(quatro otimizadores FSDP testados, todos falham; ver NOTAS e o config do 4B). A escada
0.6B/1.7B/gemma cobre a Q1; 4B e 8B ficam fora dos resultados de Q1 por não terem
treino full-parameter (limite de hardware documentado, não linha de resultado).
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
  depois (Qwen3 0.6B 6.88 -> 1.7B 5.73). A escada full-parameter para nos modelos
  que cabem em 1x L4 (0.6B/1.7B/gemma-1b); o 4B full FT não cabe nas 2x L4 (limite
  de hardware, ver NOTAS), então não tem resultado de Q1.
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
(`Qwen3-0.6B/1.7B`) e o `gemma-3-1b-it` foram só avaliados, sem fine-tuning; os
`-base`/`-pt` têm antes e depois. Tabela completa em
`results/q1_base_vs_instruct.csv`. Held-out, perplexidade (menor melhor):

| Tam. (família) | base antes | base depois (FT) | instruct sem FT |
|----------------|------------|------------------|-----------------|
| 0.6B (qwen3) | 11.47 | **6.88** | 16.30 |
| 1.7B (qwen3) | 8.59 | **5.73** | 11.92 |
| 1.0B (gemma3) | 9.57 | **5.49** | 28.21 |

(qwen3 0.6B/1.7B: par base/instruct do mesmo tamanho. gemma3: `-pt` base e `-it`
instruct, par da mesma família 1B.)

Leituras:
- **O base fine-tunado vence o instruct do mesmo tamanho com folga**: 0.6B 6.88 vs
  16.30; 1.7B 5.73 vs 11.92. O pré-treino contínuo no domínio supera o pós-treino
  de chat para esta tarefa intrínseca.
- **Domínio supera o pós-treino de chat**: o `Qwen3-0.6B-Base` fine-tunado (6.88)
  já bate o `Qwen3-1.7B` instruct sem treino (11.92), um modelo quase 3x maior; o
  pré-treino contínuo no domínio rende mais que o alinhamento de chat para esta
  tarefa intrínseca.
- **Base < instruct já no ponto de partida**: em todo tamanho, o base antes tem
  perplexidade menor que o instruct sem treino (0.6B 11.47 < 16.30; 1.7B 8.59 <
  11.92). O alinhamento de chat cobra um imposto em texto cru de diário, monotônico
  nas duas famílias (gemma-it no extremo, 28.21).
- Confirma quantitativamente, em duas famílias, a decisão de partir de modelos
  **base** nas Q1-Q3.

### Mini análise (Q1)

Três efeitos se somam e apontam na mesma direção:

1. **Adaptação de domínio supera escala.** Para perplexidade em texto de diário, o
   que mais importa não é o tamanho do modelo, é ter visto o domínio. Um base
   pequeno fine-tunado (Qwen3-1.7B-Base 5.73; gemma-3-1b-pt 5.49) bate um instruct
   maior sem treino (Qwen3-1.7B instruct 11.92). Em um orçamento de 2x L4, treinar
   um base pequeno rende mais que pegar um instruct de prateleira.
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
- **Escala ajuda** (1.7B > 0.6B em juiz e ppl). O 4B full fine-tuning não cabe nas
  2x L4 (limite de hardware documentado), então 1.7B é o maior tamanho treinado.
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

### Varredura de rank LoRA (Qwen3-0.6B, job 428)

Mesmo eval (juiz fixo Qwen3-8B, 150 itens de recall). Só muda o rank. Dados em
`results/q3_rank_sweep.csv`.

| Rank | juiz 0-5 | ppl resposta |
|------|----------|--------------|
| base | 1.49 | 9.29 |
| r4 | 1.63 | 6.57 |
| r8 | 1.66 | 6.39 |
| r16 | 1.66 | 6.29 |
| **r32** | **1.78** | 6.54 |
| r64 | 1.77 | 6.88 |

- O ganho de perplexidade satura cedo: **r8/r16 já capturam quase tudo** (~6.3-6.4).
- O juiz tem pico em **r=32** (1.78); r64 não melhora. Melhor custo-benefício: r=16-32.

### Curva de dados de SFT (Qwen3-0.6B, job 428)

Quantos pares de SFT bastam. Mesmo eval. Dados em `results/q2_data_curve.csv`.

| Pares | juiz 0-5 | ppl resposta |
|-------|----------|--------------|
| base (0) | 1.49 | 9.29 |
| 250 | 1.65 | 6.95 |
| 500 | **1.69** | 6.53 |
| 1000 | 1.49 | 6.44 |

- **Mais dados sempre baixam a perplexidade** (9.29 -> 6.44), de forma monotônica.
- O juiz, porém, **satura em ~500 pares** e oscila depois (n1000 cai ao nível do base
  no juiz apesar da menor ppl): retorno decrescente do SFT closed-book, e o juiz mede
  algo distinto da ppl teacher-forced.

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

### Comparação de professores: o professor importa? (orçamento fixo)

Ablação controlada: mesmo conjunto de 7 alunos, mesmo orçamento (400 pares de
treino por professor), mesma avaliação (recall fixo + juiz fixo Qwen3-8B); só muda
o professor que gerou os dados sintéticos. Quatro professores: Qwen3-8B, Qwen3-30B-
A3B, gemma-3-27b-it, gemma-4-31b-it. Dados em `q4_teacher_compare.csv` e
`q4_teacher_<tag>_recall.csv`. (Orçamento de 400 pares: não comparar com a tabela
Q4 principal acima, que usou 1200 pares.)

Média do juiz nos 7 alunos, por professor:

| Professor | média (7 alunos) | gemma-3-1b | qwen2.5-0.5b | qwen3-0.6b | gemma-3-270m | smollm2-360m | smollm2-135m | gpt2 |
|-----------|------------------|------------|--------------|------------|--------------|--------------|--------------|------|
| Qwen3-30B-A3B | **0.354** | 0.72 | 0.53 | 0.58 | 0.21 | 0.28 | 0.06 | 0.10 |
| gemma-4-31b-it | 0.341 | 0.71 | 0.55 | 0.47 | 0.26 | 0.20 | 0.05 | 0.15 |
| Qwen3-8B | 0.329 | 0.50 | 0.61 | 0.52 | 0.30 | 0.20 | 0.12 | 0.05 |
| gemma-3-27b-it | 0.314 | 0.63 | 0.46 | 0.49 | 0.26 | 0.22 | 0.09 | 0.05 |

Leituras:
- **O melhor professor é o Qwen3-30B**, mas a margem é pequena: o intervalo entre o
  melhor e o pior professor é 0.04 num juiz 0-5. Trocar de professor move pouco.
- **"Maior" não é monotônico**: o gemma-3-27b (27B) fica abaixo do Qwen3-8B (8B), e o
  gemma-4-31b (31B) fica acima do 8B mas abaixo do 30B. Família/qualidade do professor
  pesa mais que a contagem de parâmetros dele.
- **O aluno domina o professor**: a variação entre alunos (gemma-3-1b 0.50-0.72,
  qwen2.5-0.5b 0.46-0.61) é muito maior que a variação entre professores para um mesmo
  aluno; smollm2-135m e gpt2 ficam ~0 com qualquer professor. A capacidade do aluno em
  ler PT limita o teto, não o professor.
- **Quem mais ganha com professor forte é o gemma-3-1b** (0.50 com 8B -> 0.72 com 30B):
  um aluno já capaz aproveita melhor um professor melhor.

## Família GPT-2 (pioneira): Q1-Q4 em um modelo inglês pequeno

O GPT-2 (124M/355M/774M, BPE inglês, vocabulário 50257, sem variante instruct)
passou pelo mesmo tratamento dos outros: Q1 (pré-treino contínuo), Q2 (SFT base e
pós-Q1), Q3 (LoRA base e pós-Q1) e Q4 (aluno na destilação response-based; logit-KD
não se aplica, vocabulário difere dos teachers). Dados em `q1_gpt2.csv`, `q2_sft.csv`,
`q3_lora.csv`, `q4_distill.csv`.

Q1 (held-out, perplexidade, antes -> depois): 124M 75.2 -> 59.9; 355M 53.4 -> 40.1;
774M 44.6 -> 23.8. A adaptação de domínio funciona mesmo num modelo inglês: a
perplexidade in-domain cai muito (774M quase pela metade). E, ao contrário de
Qwen/gemma, o GPT-2 **não esquece**: o delta OOD (docentesDC) é negativo nos três
tamanhos (124M -0.82; 355M -0.37; 774M -0.99). Hipótese: partindo tão mal de
qualquer texto não-inglês, treino em script latino ajuda de forma ampla, sem o
trade-off de esquecimento visto nos modelos que já dominavam o português.

Q2/Q3/Q4 (juiz fixo Qwen3-8B, 0-5, recall do docentesDC): o juiz fica baixo em
todas as variantes. SFT pleno: 124M 0.03, 355M 0.50, 774M 0.25-0.29 (par base/Q1);
LoRA: 124M 0.11-0.13, 355M 0.09-0.33, 774M 0.20-0.21, comparável ao SFT pleno.
Destilação: juiz 0.05 (igual ao base), transfer ratio 0. Em todos os casos a
perplexidade da resposta despenca (SFT 774M 89 -> 38; destilação 1537 -> 134, -91%),
mas a nota do juiz não acompanha.

Leitura: para a métrica de linguagem (perplexidade), adaptação de domínio melhora o
GPT-2 de forma clara e sem esquecimento. Para a tarefa downstream em português
(seguir instrução, responder certo), o tokenizer inglês e a ausência de pré-treino
em PT limitam o teto: o modelo aprende a **forma** (tokens PT mais prováveis), não a
**tarefa**. Funciona como piso/referência negativa que dimensiona o quanto a escolha
de um base já competente em português (Qwen/gemma) importa.

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

### Motor 30B (Qwen3-30B-A3B-Instruct-2507-FP8)

O 30B-A3B-FP8 preenche as 2 L4 (device_map=auto), então o juiz aqui e o **proprio
motor** (auto-julgamento), nao o juiz fixo 8B: estes numeros nao sao diretamente
comparaveis a tabela acima (a referencia cross-engine continua sendo o juiz fixo 8B).
Mesmo benchmark de 30 perguntas. Dados em `results/q5_rag_30b.csv` (job 439).

| Motor (auto-juiz) | baseline | standard | agentic_graph |
|-------------------|----------|----------|---------------|
| Qwen3-30B-A3B-FP8 | 1.57 | **3.03** | 2.90 |

Leituras:
- **RAG ajuda mesmo o motor maior**: standard sobe +1.47 sobre o baseline (1.57 ->
  3.03), o mesmo padrao dos motores menores: a recuperacao e o ganho principal.
- **Grafo nao separa do standard** (2.90 vs 3.03), reforcando a conclusao geral: nesta
  tarefa de achar um fato em texto, o grafo/multi-hop agrega pouco. A auto-correcao do
  agente disparou em 3/30 casos.
- O baseline do 30B (1.57) ja e maior que o do 8B (1.10 no juiz fixo), coerente com um
  motor mais capaz closed-book, mas o teto com RAG fica proximo (3.03 vs 3.50 do 8B no
  corpus balanceado): a recuperacao nivela motores fortes.

### Students destilados como motor RAG (Q4 -> Q5)

Pergunta: um aluno destilado pequeno e barato pode servir de motor de geracao no RAG?
Cada student da Q4 rodou como motor sobre o mesmo indice e benchmark de 30 perguntas,
juiz fixo Qwen3-8B (comparavel a tabela de motores acima). Dados em `q5_engines.csv`
(`kind=distill-student`) e `q5_student_*.csv`.

| Motor (student destilado) | baseline | standard | agentic_graph |
|---------------------------|----------|----------|---------------|
| qwen2.5-0.5b-distill | 0.07 | **3.87** | 3.53 |
| gemma-1b-distill | 0.73 | 1.30 | 1.33 |
| qwen3-0.6b-distill | 0.83 | 0.83 | 1.10 |
| smollm2-360m-distill | 0.17 | 0.87 | 1.17 |
| smollm2-135m-distill | 0.00 | 0.47 | 0.33 |
| gpt2-distill | 0.00 | 0.00 | 0.00 |

Leituras:
- **Sim, e bem**: o `qwen2.5-0.5b-distill` salta de 0.07 closed-book para 3.87 com RAG
  standard (ganho +3.80), acima do proprio Qwen3-8B standard (2.70) no mesmo juiz fixo.
  Um motor 16x menor, quase nulo sem contexto, vira competitivo quando le a evidencia
  recuperada: a recuperacao carrega o resultado, nao o tamanho do motor.
- **O ganho exige que o motor use o contexto**: o `qwen3-0.6b-distill` parte do maior
  baseline (0.83) mas nao melhora com standard (0.83), provavelmente respondendo do
  proprio conhecimento destilado e subutilizando a recuperacao; o grafo o ajuda de leve
  (1.10). Capacidade de seguir o contexto importa mais que conhecimento previo.
- **Piso ingles**: o `gpt2-distill` fica em 0.00 em todos os modos. Sem PT no
  tokenizer/pre-treino, nem o RAG resgata; confirma o limite visto na secao GPT-2.
- Escala dos students nao ordena o resultado (135M < 360M, mas 0.5B > 0.6B > 1B aqui):
  o que separa e a familia/qualidade do aluno em ler PT, nao a contagem de parametros.

### Motores grandes: gemma-3-27b-it e gemma-4-31b-it (4-bit)

Os dois gemma grandes como motor RAG (4-bit NF4, fixados no GPU0; juiz fixo Qwen3-8B
no cuda:1), mesmo indice e benchmark. Dados em `q5_engine_gemma-3-27b-it.csv` e
`q5_engine_gemma-4-31b-it.csv`.

| Motor (4-bit) | baseline | standard | agentic_graph |
|---------------|----------|----------|---------------|
| gemma-3-27b-it | 1.30 | **3.10** | 2.97 |
| gemma-4-31b-it | 1.10 | (OOM, ver abaixo) | (OOM) |

Leituras:
- **Um gemma grande e motor RAG forte**: o `gemma-3-27b-it` em standard (3.10) supera o
  `Qwen3-8B` (2.70) no mesmo juiz fixo, confirmando que motor mais capaz ajuda, mas o
  ganho continua vindo da recuperacao (+1.80 sobre o baseline 1.30).
- **Limite de hardware no 31B**: o `gemma-4-31b-it` (4-bit ~16 GB) nao cabe de forma
  confiavel em uma L4 junto do juiz 8B quando ha contexto recuperado: o baseline (sem
  RAG) roda limpo (1.10), mas o standard sofre OOM em ~11-17 das 30 perguntas mesmo com
  geracoes de 256 tokens, e o agentic_graph (contexto acumulado) nao fecha. Nas
  perguntas que couberam o standard fica ~3.36 (na faixa do 27b), entao a limitacao e de
  memoria, nao de qualidade. Para um numero limpo do 31b seria preciso uma avaliacao em
  dois passos (gerar com o motor nos dois GPUs, depois julgar), nao feita aqui.

### Métricas de recuperação (hit-rate@k do retriever)

Isola o retriever do gerador: mede se a evidência chega ao prompt. Como o benchmark
não tem id de documento gold, usa-se o proxy padrão de **answer hit-rate@k** (a resposta
esperada aparece em algum dos k chunks recuperados). Embedder bge-m3, 30 perguntas.
Dados em `results/q5_retrieval.csv` (script `scripts/eval_retrieval.py`).

| Método | hit@1 | hit@3 | hit@5 | hit@10 |
|--------|-------|-------|-------|--------|
| plain (similaridade) | 0.43 | 0.57 | **0.60** | 0.63 |
| MMR | 0.43 | 0.47 | 0.47 | 0.57 |
| plain - factual | 0.50 | 0.67 | 0.67 | 0.72 |
| plain - multihop | 0.33 | 0.42 | 0.50 | 0.50 |

Leituras:
- **A recuperação simples (plain) supera a MMR** em hit-rate: o MMR troca relevância por
  diversidade e, nesta tarefa de achar um fato pontual, isso tira chunks certos do topo.
- **Factual > multihop** (hit@5 0.67 vs 0.50): perguntas multi-hop espalham a evidência
  por mais de um documento, mais difícil de cobrir no top-k.
- **Teto em ~0.63 (hit@10)**: em ~37% das perguntas a resposta nunca aparece nos chunks
  recuperados (resposta fora do subconjunto indexado, ou fraseada de outro jeito, ou
  miss do retriever). Isso limita o teto do gerador e explica parte do erro do RAG.

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

## Síntese transversal (cruzando as questões)

Leituras que só aparecem ao cruzar os resultados de várias questões (todas no notebook):

- **Pós-treino, leaderboard justo (docentes recall, n=150).** No mesmo conjunto e juiz,
  o topo é LoRA: `Qwen3-1.7B` LoRA iniciado em Q1 (2.11) > LoRA base (2.05) > SFT pleno
  de Q1 (1.99). LoRA iguala ou supera o SFT pleno em 5 de 6 pares treinando ~1.7% dos
  parâmetros: a fronteira custo x qualidade favorece o PEFT.
- **RAG, contribuição pareada por pergunta (motor 8B, standard vs baseline, n=30).** Em
  14 perguntas o RAG melhora, 12 empatam e em 4 piora; ganho médio de +1.60 no juiz. O
  RAG não é uniformemente positivo, mas o saldo é claramente a favor.
- **RAG por tipo de pergunta.** O ganho do `standard` vem das **factuais** (baseline 0.5
  -> standard 3.00); nas **multi-hop** o standard quase não move (2.0 -> 2.25), e é o
  modo **agentic_graph** que ajuda justamente nelas (2.92 > 2.25 do standard). Refina a
  conclusão geral: o grafo/multi-hop agrega pouco na média porque a maioria das
  perguntas é factual, mas no subconjunto multi-hop ele é o melhor modo.
- **Escala não ordena o RAG.** O melhor motor no juiz fixo é um aluno destilado de 0.5B
  (`qwen2.5-0.5b-distill` standard 3.87), acima do `gemma-3-27b-it` (3.10) e do
  `Qwen3-8B` (2.70): com a evidência recuperada no prompt, ler o contexto pesa mais que
  o tamanho do motor.
- **Mapa família x tarefa.** Cruzando o melhor juiz por família em Q2/Q3/Q4 (docentes) e
  Q5 (RAG): qwen3 lidera o pós-treino closed-book; no RAG o que decide é a família saber
  ler PT e usar o contexto (gpt2 fica em 0 em tudo, piso inglês). A escolha de um base já
  competente em português domina o resultado em todas as frentes.

## Convenção de colunas (runs.csv)

`date, question, model, params, variant (base|instruct|vlm), modality (text|vlm),
method, dataset, eval_set, ppl_before, ppl_after, ce_before, ce_after,
tokacc_before, tokacc_after, train_loss, blocks, gpus, config, notes`.
