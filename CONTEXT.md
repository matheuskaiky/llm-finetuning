# CONTEXT.md - Contexto Geral do Projeto

> Documento de **alto nível**: visão de produto, escopo macro e cronograma.
> Para o contexto técnico (arquitetura, código, contratos), consulte
> [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

## 1. Visão do produto

Construir o **ciclo de vida completo de um LLM** especializado em dados públicos
do Piauí. O modelo é treinado e avaliado sobre dois conjuntos de dados oficiais
da disciplina e, ao final, é exposto por meio de uma aplicação de RAG com camada
de segurança (guardrails).

A entrega tem caráter acadêmico (disciplina **Tópicos em IA**, UFPI-CCN-DC,
Prof. Raimundo Moura, 2026.1), mas é estruturada como um projeto de engenharia
real: código desacoplado, configurável e reprodutível.

### Datasets

| Dataset | Conteúdo | Uso principal |
|---------|----------|---------------|
| `diariosPrefeituras` | Diários oficiais das prefeituras (PDF -> `.txt`). | Pré-treino contínuo (Q1) e RAG (Q5). |
| `docentesDC` | Dados de docentes do Departamento de Computação (`.txt`). | Geração de pares Q&A para SFT/LoRA (Q2, Q3) e RAG (Q5). |

> **Artefatos compartilhados (ver `tarefa.md`).** Alguns itens são de uso geral,
> produzidos uma única vez pelo grupo responsável e consumidos por todos. O nosso
> grupo **não os recria**: apenas os consome e contribui no formato pedido.
>
> | Artefato | Grupo responsável | Nossa obrigação |
> |----------|-------------------|-----------------|
> | Benchmark de P&R sobre a UFPI | Grupo 03 | repassar informações; consumir o benchmark |
> | Dataset unificado `diariosPrefeituras` (.txt) | Grupo 01 | baixar nossa parcela de PDFs; consumir o dataset |
> | Dataset unificado `docentesDC` (.txt) | Grupo 08 | repassar dados no formato pedido; consumir o dataset |
>
> O que construímos por conta própria são os datasets/benchmarks específicos de
> cada questão (ex.: os >= 1.000 pares de SFT, os benchmarks de Q1/Q4/Q5/Q6).

## 2. Escopo macro (as 6 frentes)

| # | Frente | Entregável macro |
|---|--------|------------------|
| Q1 | **Pré-treino contínuo** | Treinar um LLM base sobre os diários e avaliar antes/depois (perplexidade, entropia cruzada, acurácia de previsão de tokens) com benchmark >= 25 perguntas. |
| Q2 | **Pós-treino (SFT)** | Gerar >= 1.000 pares `{instruction, input?, output}` do `docentesDC` e fazer fine-tuning supervisionado; avaliar antes/depois. |
| Q3 | **Pós-treino (LoRA/QLoRA)** | Repetir o experimento com PEFT (LoRA e/ou QLoRA); comparar com o SFT pleno. |
| Q4 | **Destilação de conhecimento** | Definir teacher e student, destilar via dataset sintético; benchmark de 100 perguntas; analisar transferência de conhecimento. |
| Q5 | **RAG** | Aplicação RAG (Standard, Agentic ou Self-Reflective) sobre os datasets; benchmark de 30 perguntas; medir contribuição do RAG. |
| Q6 | **Guardrails** | Camada de proteção (bloqueio/reescrita/classificação/mascaramento); benchmark de 30 perguntas; medir grau de proteção adicionado. |

> Cada grupo deve escolher um **LLM diferente** (mesma família ou famílias
> distintas). Quando possível, comparar mais de um modelo com tamanhos de
> parâmetros diferentes.

## 3. Princípios do projeto

- **Desacoplamento ambiente local <-> nuvem.** O alvo primário é GPU local/servidor,
  mas toda a execução é dirigida por configuração (YAML), permitindo migrar para a
  nuvem sem reescrever código.
- **SOLID, com ênfase em OCP.** Novos modelos, métodos de treino e métricas entram
  por **extensão** (novas classes/configs), sem modificar o núcleo.
- **Reprodutibilidade.** Seeds fixas, configs versionadas e benchmarks salvos.
- **Idioma.** Código, funções e classes em inglês; documentação de contexto
  (`CONTEXT.md`, `PROJECT_CONTEXT.md`) e issues em português.

## 4. Cronograma de entregas

A equipe tem **4 integrantes** (P1-P4). As tarefas detalhadas estão nas *issues*
do GitHub, agrupadas pelos marcos abaixo.

### Marco 1 - 11/06/2026 - 1ª Apresentação Parcial (Setup + MVP)
Foco em fundação e prova de conceito:
- Estrutura do repositório, ambiente Python e tooling.
- Abstrações de núcleo (interfaces + registry + config YAML).
- Ingestão de dados MVP: download dos PDFs e conversão PDF -> `.txt`.
- Avaliação MVP: carregar 1 LLM e medir perplexidade/entropia/acurácia de token
  em um benchmark mínimo (baseline pré-treino).

### Marco 2 - 25/06/2026 - 2ª Apresentação Parcial (Features core)
Foco nas funcionalidades centrais/avançadas:
- Q1 - Pré-treino contínuo + benchmark >= 25 + avaliação antes/depois.
- Q2 - Geração de >= 1.000 pares Q&A + SFT + avaliação.
- Q3 - LoRA/QLoRA como estratégia de treino + comparação.
- Q5 - RAG (um tipo) + benchmark de 30 + análise de contribuição.

### Marco 3 - 07/07/2026 - Apresentação e Entrega Final
Foco em integração, polimento e correção de bugs:
- Q4 - Destilação teacher->student + benchmark de 100 + análise de transferência.
- Q6 - Guardrails + benchmark de 30 + análise de proteção.
- Integração end-to-end (CLI/config unificada).
- Relatório final, polimento, correção de bugs e checagem de reprodutibilidade.

## 5. Formato da entrega final

> Confirmar com o Prof. Raimundo Moura se há um formato obrigatório (ex.: PDF do
> relatório, prazo de envio, plataforma). O descrito abaixo é o padrão que
> adotamos por conta própria; ajustar se o professor exigir algo diferente.

A entrega final (07/07) é composta por **quatro partes**: o repositório, o
relatório técnico, os artefatos de cada experimento e a apresentação. A nota vem
da reprodutibilidade e da clareza da análise antes/depois, não só de "rodar".

### 5.1. Artefatos entregues

| Artefato | Onde | Observação |
|----------|------|------------|
| Código (pacote, scripts, testes) | repositório GitHub | fonte principal; `main` no estado final. |
| Relatório técnico | `docs/RELATORIO.md` (e PDF exportado) | metodologia + resultados + análise por questão (Q1-Q6). |
| Notebooks de demonstração | `notebooks/` | um por questão (ou um geral), reproduzindo os experimentos. |
| Datasets gerados | `data/` (local) + links | pares Q&A (Q2), dataset sintético (Q4); grandes, fora do Git. |
| Benchmarks | `benchmarks/*.jsonl` | Q1 (>=25), Q4 (100), Q5 (30), Q6 (30), versionados. |
| Resultados de avaliação | `benchmarks/results/*.json` + tabelas no relatório | JSON bruto (gerado) e tabela consolidada no relatório. |
| Checkpoints / adapters | HuggingFace Hub ou Drive | pesos são grandes; hospedar fora e referenciar por link. |
| Slides da apresentação | `docs/` | material do dia 07/07. |

Os pesos de modelo e os datasets brutos **não** vão para o Git (ver `.gitignore`).
São hospedados externamente (HuggingFace Hub de preferência) e linkados no
relatório, com a versão/commit que os gerou.

### 5.2. Formato das saídas de cada experimento

Toda avaliação produz um JSON padronizado (gerado por `save_results`) com as
métricas **antes e depois**. Schema sugerido:

```json
{
  "question": "Q1",
  "model": "gpt2",
  "method": "continual_pretrain",
  "dataset": "diariosPrefeituras",
  "before": {"perplexity": 0.0, "cross_entropy": 0.0, "token_accuracy": 0.0},
  "after":  {"perplexity": 0.0, "cross_entropy": 0.0, "token_accuracy": 0.0},
  "config": "configs/pretrain_diarios.yaml",
  "seed": 42
}
```

No relatório, cada questão vira uma **tabela comparativa antes/depois** (e um
gráfico de barras quando ajudar). As métricas por questão:

- **Q1 (pré-treino):** perplexidade, entropia cruzada, acurácia de previsão de
  tokens (antes/depois) sobre o benchmark dos diários.
- **Q2 (SFT) e Q3 (LoRA/QLoRA):** as mesmas métricas intrínsecas + exemplos
  qualitativos de respostas; Q3 também compara custo (VRAM, tempo) com o SFT.
- **Q4 (destilação):** tabela teacher x student (antes/depois) no benchmark de
  100; conclusão explícita sobre houve ou não transferência de conhecimento.
- **Q5 (RAG):** comparação com e sem RAG no benchmark de 30 (acurácia/qualidade
  e aderência ao contexto); análise do grau de contribuição.
- **Q6 (guardrails):** % de respostas nocivas bloqueadas x % de respostas
  legítimas preservadas no benchmark de 30; conclusão sobre o grau de proteção.

### 5.3. Estrutura do relatório (uma seção por questão)

1. **Objetivo** - o que a questão pede.
2. **Modelo(s) e dataset** - qual LLM, qual dataset, tamanhos.
3. **Metodologia** - como foi treinado/configurado (hiperparâmetros principais).
4. **Resultados** - tabela antes/depois + gráfico, com a métrica de cada questão.
5. **Análise** - o que os números mostram, limitações, surpresas.
6. **Reprodução** - comando exato e a config usada (ex.: `python scripts/train.py
   --config configs/pretrain_diarios.yaml`).

### 5.4. Reprodutibilidade (critério de qualidade)

- Seed global fixa (`set_global_seed`); toda config versionada em `configs/`.
- Cada resultado do relatório aponta para o commit, a config e o comando que o
  gerou.
- Versões de dependências congeladas (`pyproject.toml`; opcionalmente um lock).
- Qualquer integrante consegue refazer um experimento a partir da config.

### 5.5. O que demonstrar em cada marco

- **11/06 (parcial):** repositório montado e o MVP rodando (ingestão de dados +
  uma avaliação baseline de um LLM).
- **25/06 (parcial):** Q1, Q2, Q3 e Q5 com resultados parciais e tabelas
  antes/depois preenchidas.
- **07/07 (final):** as seis questões integradas, relatório consolidado, slides e
  reprodutibilidade conferida.
