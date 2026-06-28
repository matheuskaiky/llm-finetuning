# Guia para escrever o artigo (git-ignorado)

> Documento de método para transformar os resultados do projeto em um artigo. Hoje o
> projeto tem muitos números (ver `results/README.md` e o notebook), mas pouca
> análise por cima deles: estatística de variação, validação do instrumento de
> medida (o juiz), correlações e análise de erro. Este guia diz o que escrever, que
> análises acrescentar e onde cada peça entra. Não é o relatório; é o plano dele.
> O relatório final vai para `docs/RELATORIO.md` (só iniciar quando o autor liberar).

## 1. O problema atual: dados sem estudo

O que já existe é forte em cobertura: 6 questões, várias famílias e tamanhos,
ablações, antes/depois. O que falta é a camada de análise que separa um relatório de
um artigo:

1. **Toda métrica é estimativa pontual de uma única corrida (seed 42).** Não há
   intervalo de confiança nem múltiplas seeds. Uma média de juiz 0-5 sobre 30-150
   itens tem variância amostral que não é reportada. Sem isso, deltas como "+0.20" ou
   "5 de 6 casos" podem estar dentro do ruído.
2. **O instrumento de medida não foi validado.** O juiz é um LLM (Qwen3-8B). Os
   próprios resultados já mostraram artefato de auto-julgamento (gemma-it inflava),
   mas não há calibração do juiz contra rótulo humano. Sem isso, todos os números do
   juiz herdam o viés do juiz.
3. **Poucas relações entre variáveis.** Há scatter de escala vs qualidade, mas sem
   coeficiente/ajuste; sem regressão de transfer ratio vs capacidade do aluno; sem
   teste pareado nas comparações (LoRA vs SFT, RAG vs baseline).
4. **Análise de erro é só anedótica.** Há exemplos qualitativos no RAG, mas não uma
   taxonomia de falhas que sustente a discussão.

O artigo nasce de fechar essas quatro lacunas. As seções 4 e 5 abaixo dizem como.

## 2. Tese central e perguntas de pesquisa

O fio condutor que amarra as 6 questões (e dá uma tese, não só um relatório de
atividades):

> **Sob um orçamento pequeno de GPU (2x L4, 22 GB), adaptação ao domínio e
> recuperação rendem mais que escala; PEFT alcança o full fine-tuning; e na
> destilação e no RAG o que decide é o uso do contexto/idioma, não o tamanho.**

Cada questão vira uma pergunta de pesquisa (PP) testável:

- **PP1 (Q1):** o pré-treino contínuo no domínio melhora as métricas intrínsecas, e
  por quanto, comparado a usar um instruct de prateleira? (adaptação vs escala vs
  alinhamento de chat)
- **PP2 (Q2):** o SFT melhora a resposta no domínio, e o ganho depende de quão fraco
  o base era? Q1 e Q2 se somam?
- **PP3 (Q3):** PEFT (LoRA/QLoRA) iguala o SFT pleno a uma fração do custo? E
  destrava modelos maiores que o full-FT não alcança? (ver lacuna do 8B/9B no roadmap)
- **PP4 (Q4):** há transferência de conhecimento professor->aluno, e o que limita o
  teto, o professor ou o aluno?
- **PP5 (Q5):** qual a contribuição marginal da recuperação, e o grafo/multi-hop
  agrega sobre a recuperação simples?
- **PP6 (Q6):** quanto de proteção a camada adiciona, e quão robusta ela é a
  reformulações?

Escreva a Introdução em torno da tese e das PPs, não em torno de "implementamos X".

## 3. Estrutura do artigo (IMRaD) e onde cada peça entra

- **Resumo/Abstract:** problema (ciclo de vida de LLM para texto público em PT sob
  GPU pequena), o que foi feito, o achado principal (1-2 números), a limitação chave.
- **Introdução:** contexto (domínio público do Piauí, restrição de hardware como
  tema), as 6 PPs, contribuições. Fechar com o mapa do artigo.
- **Trabalhos relacionados:** pré-treino contínuo, SFT, LoRA/QLoRA, destilação, RAG,
  guardrails. Usar `docs/references.bib` (já existe) e amarrar cada técnica à PP.
- **Metodologia:** dados e proveniência (diários `dom-pi-corpus-2025`, docentesDC;
  ressalva de gabarito gerado por IA), modelos e por que base e não instruct,
  protocolo de avaliação (held-out disjunto, recall in-domain, juiz fixo 0-5, ppl
  teacher-forced), hardware (2x L4) e regimes de treino. Material pronto em
  `EXPLICACAO_PROFESSOR.md`, `PROJECT_CONTEXT.md` e `NOTAS.md`.
- **Resultados:** uma subseção por questão, cada uma respondendo a sua PP, com as
  análises da seção 4 (não só as tabelas atuais).
- **Discussão:** a síntese transversal (já esboçada em `results/README.md` e na seção
  6 do notebook): adaptação > escala; PEFT ~ full; aluno domina professor;
  recuperação é o ganho; regra é frágil. Aqui entram as correlações e a leitura.
- **Limitações e ameaças à validade:** seção 6 abaixo.
- **Conclusão e trabalhos futuros:** puxar do roadmap (QLoRA 8B/9B, escala da Q1,
  professor com RAG, roteador, classificador de guardrail, gabarito a mão).

## 4. Análises a acrescentar (o miolo do estudo)

Estas são as análises que faltam para os números virarem evidência. Ordenadas por
retorno. A maioria roda sobre os CSVs e os `*_details.jsonl` já existentes, sem
re-treinar nada.

### 4.1 Variação e significância (transversal, P0 do artigo)

- **Intervalos de confiança por bootstrap** para toda média de juiz e de ppl. Os
  itens estão nos `results/*_recall_details.jsonl` (nota por item); reamostrar com
  reposição (ex.: 10000 vezes) dá um IC95%. Sem re-treino. Põe barra de erro em todo
  gráfico de barras do notebook.
- **Testes pareados** onde a comparação é no mesmo conjunto de itens: LoRA vs SFT
  pleno (mesmos 150 de recall), RAG standard vs baseline (mesmos 30, já pareado na
  seção 4.3 do notebook), QLoRA vs LoRA. Usar bootstrap pareado ou Wilcoxon signed-
  rank sobre as diferenças por item. Transforma "LoRA vence em 5 de 6" em "delta
  médio X, IC95% [a, b], p=...".
- **Múltiplas seeds, ao menos no caso central.** Reportar tudo com seed única é a
  fraqueza mais fácil de atacar. Se o orçamento não permitir reseedar tudo, rodar 3
  seeds no experimento âncora (ex.: SFT 0.6B e LoRA 0.6B) e mostrar que a ordem se
  mantém. Documentar como ameaça à validade o que não foi reseedado.

### 4.2 Validação do juiz (transversal, P0 do artigo)

O juiz é o instrumento; sem calibrá-lo, todo número 0-5 é suspeito.

- **Concordância com humano.** Amostrar ~50 itens (estratificados por questão e por
  faixa de nota), rotular à mão (0-5) e medir concordância juiz vs humano: correlação
  de Spearman, erro absoluto médio e, binarizando "aceitável/não", kappa de Cohen.
  Liga-se à pendência do gabarito feito a mão (roadmap P1).
- **Sensibilidade ao juiz.** Repetir um subconjunto com um segundo juiz (ex.: um
  gemma grande) e ver se o ranking entre modelos se preserva. Já há sinal de que o
  juiz importa (auto-julgamento do gemma-it); quantificar.
- **Reportar a escala honestamente:** o teto observado é baixo (teacher 8B marca 0.66
  no recall difícil). Discutir que o juiz mede aderência a uma resposta de referência
  que também veio de IA (circularidade), não verdade factual independente.

### 4.3 Relações entre variáveis (por questão)

- **Q1, escala vs perplexidade:** ajustar perplexidade depois vs log(params) e
  reportar a inclinação; testar se família (Qwen vs gemma vs gpt2) muda o intercepto.
  Hoje a seção 1.4 do notebook mostra os pontos sem ajuste.
- **Q4, transferência vs capacidade do aluno:** regressão de transfer ratio (ou juiz
  destilado) vs juiz-base do aluno. A leitura "o aluno domina o professor" fica forte
  se vier de um coeficiente, não de olhar a tabela.
- **Q3, custo vs qualidade:** Pareto de qualidade (juiz) vs parâmetros treinados (LoRA
  ~1.7%) e vs memória/tempo. Com a linha QLoRA 8B/9B, vira o gráfico-chave do PEFT.
- **Q5, teto do retriever vs acerto do gerador:** correlacionar hit-rate@k (já em
  `q5_retrieval.csv`) com a nota do juiz por pergunta. Mostra quanto do erro do RAG é
  recuperação vs geração.

### 4.4 Análise de erro estruturada (por questão, sustenta a Discussão)

- **RAG (Q5):** dos `*_details`/`q5_qualitativos.md`, classificar as falhas em
  categorias (alucinação, recusa, recuperação errou, pergunta ambígua, resposta certa
  mas juiz penalizou). Contar a frequência de cada uma. Já há o material qualitativo;
  falta a taxonomia quantificada.
- **Guardrails (Q6):** detalhar os falsos negativos do teste parafraseado (quais
  reformulações passam) para fundamentar a conclusão de fragilidade da regra.
- **Q2/Q3:** olhar onde a ppl cai mas o juiz não sobe (efeito visto na curva de
  dados): exemplos de respostas fluentes porém factualmente erradas (closed-book).

### 4.5 Fechar a lacuna de modelo grande (Q3)

Ver o roadmap: rodar **QLoRA num 8B/9B** e incluir na análise de custo x qualidade
(4.3). É a peça que falta para a tese "PEFT destrava o modelo grande sob 2x L4" e
provavelmente o experimento de maior impacto que ainda cabe no hardware.

## 5. Plano de figuras e tabelas

Aproveitar o notebook (já tem as 6 seções), mas cada figura precisa de uma frase de
leitura e, onde fizer sentido, barra de erro (seção 4.1).

- F1 (Q1): perplexidade antes/depois por modelo, com base fine-tunado vs instruct sem
  treino lado a lado (a evidência mais limpa do artigo).
- F2 (Q1): escala vs perplexidade, com reta ajustada por família (seção 4.3).
- F3 (Q2/Q3): leaderboard do pós-treino com IC95%; destacar LoRA >= SFT pleno.
- F4 (Q3): Pareto qualidade vs custo (parâmetros treinados / memória), com o ponto
  QLoRA 8B/9B.
- F5 (Q4): transfer ratio por aluno + reta vs capacidade-base (seção 4.3).
- F6 (Q4): heatmap professor x aluno (já existe, seção 3.3).
- F7 (Q5): contribuição do RAG pareada por pergunta + por tipo (factual vs multi-hop).
- F8 (Q5): hit-rate@k vs nota do juiz.
- F9 (Q6): barras com vs sem guardrail por categoria, e a queda no teste parafraseado.
- T1: tabela-resumo das 6 questões (1 linha por questão: PP, métrica, antes, depois,
  delta com IC). É a tabela que um avaliador lê primeiro.

## 6. Limitações e ameaças à validade (escrever explicitamente)

Antecipar as críticas fortalece o artigo:

- **Circularidade do gabarito.** Perguntas/respostas de referência geradas com IA e
  juiz LLM: o gabarito herda viés do gerador. Mitigação proposta: gabarito a mão
  (roadmap P1) e validação do juiz (4.2).
- **Seed única / sem variância reportada** na maioria das corridas (atacar com 4.1).
- **Teto de hardware.** 4B+ em full-parameter não treinado (Q1/Q2); 31B em RAG com
  OOM. São limites de recurso, não de método; deixar isso claro para não confundir
  com resultado negativo.
- **Snapshot único do córpus** (2.000 docs na Q1, 1 época): mede direção e magnitude
  do efeito, não a perplexidade absoluta de um treino completo (trabalho futuro).
- **Domínio e idioma específicos** (texto público do Piauí, PT-BR): generalização
  para outros domínios não é testada.
- **Guardrails por regra** são frágeis a paráfrase (já medido); não é robustez
  adversarial completa.

## 7. Checklist antes de submeter

- [ ] Cada número no texto bate com o CSV-fonte (a fonte é o CSV, não a memória).
- [ ] Toda média de juiz/ppl tem n e IC95%.
- [ ] Toda comparação "A > B" tem teste pareado e tamanho de efeito, não só a média.
- [ ] O juiz foi validado contra ao menos uma amostra humana.
- [ ] Cada figura tem uma frase de leitura e unidades/escala explícitas.
- [ ] Seeds, versões de biblioteca, ids de modelo/dataset e hiperparâmetros estão no
      apêndice (puxar do `NOTAS.md`).
- [ ] Limitações da seção 6 estão escritas, não implícitas.
- [ ] As contribuições da Introdução são respondidas na Conclusão.

## 8. Fontes internas para cada seção

- Métodos e decisões: `EXPLICACAO_PROFESSOR.md`, `PROJECT_CONTEXT.md`, `NOTAS.md`.
- Números e leituras por questão: `results/README.md` e `notebooks/graficos_resultados.ipynb`.
- Prompts (juiz/gerador/crítico/extrator): `docs/PROMPTS.md`.
- Quantidades e proveniência dos dados: `docs/PERGUNTAS.md` e a tabela de origem das
  perguntas em `results/README.md`.
- Pendências e trabalhos futuros: `docs/ROADMAP_GERAL.md` e `docs/RAG_ROADMAP.md`.
- Referências bibliográficas: `docs/references.bib`.
