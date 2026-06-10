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
