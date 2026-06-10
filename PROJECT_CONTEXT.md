# PROJECT_CONTEXT.md - Contexto Técnico

> Documento **estritamente técnico**. Mapeia a arquitetura, a árvore de
> diretórios, os contratos das interfaces e funções complexas.
> **Regra do projeto:** consulte e **atualize este arquivo antes de qualquer
> alteração estrutural** (criar/remover pastas, módulos ou interfaces).
>
> Para a visão de produto e o cronograma, veja [`CONTEXT.md`](CONTEXT.md).

## 1. Stack tecnológica

| Camada | Ferramenta | Papel |
|--------|------------|-------|
| Runtime | Python 3.12, PyTorch | Base de execução e tensores. |
| Modelos/treino | `transformers`, `datasets`, `accelerate` | Carregamento de modelos, datasets e orquestração de treino. |
| PEFT | `peft`, `bitsandbytes` | LoRA/QLoRA e quantização (Q3). |
| SFT | `trl` (`SFTTrainer`) | Fine-tuning supervisionado (Q2). |
| Destilação | custom (loss KL sobre HF Trainer) | Teacher->student (Q4). |
| RAG | `langchain`/`llama-index` + `faiss`/`chromadb` + `sentence-transformers` | Recuperação + geração (Q5). |
| Guardrails | `nemoguardrails`/`guardrails-ai` + camada custom | Camada de segurança (Q6). |
| Avaliação | métricas custom + opcional `lm-eval-harness` | Perplexidade, entropia cruzada, acurácia de token, benchmarks. |
| Dados | `pypdf`/`pdfplumber` | PDF -> `.txt`. |
| Config | `hydra-core`/`pydantic-settings` + YAML | Execução dirigida por configuração (base do OCP). |
| Tracking | `wandb`/TensorBoard | Comparação de métricas antes/depois. |
| Qualidade | `pytest`, `ruff`, `pre-commit` | Testes e estilo. |

## 2. Árvore de diretórios (nível macro)

```
llm-finetuning/
├── README.md            # (EN) Apresentação do projeto.
├── CONTEXT.md           # (PT) Visão de produto, escopo macro e cronograma.
├── PROJECT_CONTEXT.md   # (PT) Este arquivo - contexto técnico.
├── CLAUDE.md            # (EN) Regras do agente de IA - não versionado (.gitignore).
├── tarefa.md            # Enunciado original - local, não versionado (.gitignore).
├── .gitignore
├── pyproject.toml       # Dependências + config de ruff/pytest.
├── requirements-dev.txt # Subconjunto leve de deps para testes sem o stack de ML.
├── src/llm_finetuning/  # Código-fonte (EN), pacote único. Núcleo + módulos das 6 questões.
├── configs/             # Configuração YAML (modelos/métodos/ambiente). Habilita o OCP.
├── data/                # Datasets brutos e processados. NÃO versionado.
├── benchmarks/          # Conjuntos de perguntas de avaliação (Q1>=25, Q4=100, Q5=30, Q6=30).
├── notebooks/           # Experimentos exploratórios (Jupyter).
├── scripts/             # Entrypoints de CLI (train/eval/rag/...).
├── tests/               # Testes (pytest).
└── docs/                # Documentação adicional (relatórios, figuras, decisões).
```

> **Propósito de cada pasta principal**
> - **`src/`** - todo o código de produção. Subdividido internamente conforme o
>   blueprint da seção 3 (as subpastas serão criadas sob demanda, à medida que
>   cada módulo nascer - evitamos microsubpastas vazias no início).
> - **`configs/`** - descreve *o que* rodar (modelo, método, ambiente) sem mudar o
>   código. É o ponto de extensão central (OCP): um novo experimento = nova config.
> - **`data/`** - datasets locais (ignorados pelo Git por tamanho/licença).
> - **`benchmarks/`** - perguntas + respostas de referência usadas na avaliação.
> - **`notebooks/`** - exploração rápida; código estável migra para `src/`.
> - **`scripts/`** - finos invólucros de CLI que apenas leem uma config e chamam `src/`.
> - **`tests/`** - testes unitários/integração das abstrações e utilitários.
> - **`docs/`** - material de apoio das apresentações e decisões de arquitetura.

## 3. Arquitetura (SOLID / OCP)

**Princípio central:** interfaces estáveis + implementações plugáveis resolvidas
por um *registry/factory* a partir da configuração YAML. Trocar de modelo, método
de treino ou ambiente (local <-> nuvem) é feito por **extensão/configuração**, sem
modificar o núcleo.

### 3.1. Camadas dentro de `src/llm_finetuning/`

Marcadas com [x] as já implementadas (Marco 1); as demais nascem sob demanda.

```
src/llm_finetuning/
├── core/          # [x] Interfaces (ABCs), registry/factory, config loader, seed.
├── data/          # [x] DatasetLoader: PdfToTextLoader (PDF->txt), TextCorpusLoader. (Q&A, splits a seguir.)
├── models/        # [x] ModelProvider: LocalModelProvider, CloudModelProvider (placeholder).
├── evaluation/    # [x] Evaluator + Metrics (perplexidade, entropia, acurácia de token).
├── training/      # [~] Trainers (Strategy): ContinualPretrainTrainer (Q1). SFT/LoRA/distill a seguir.
├── rag/           # [ ] RagPipeline: retriever + generator (Standard/Agentic/Self-Reflective).
└── guardrails/    # [ ] GuardrailLayer: filtros de entrada/saída componíveis.
```

Componentes registram-se nos registries de `core/registry.py`
(`MODEL_PROVIDERS`, `DATASET_LOADERS`, `TRAINERS`, `METRICS`, `EVALUATORS`) e são
resolvidos por `instantiate(registry, ComponentSpec)` a partir do YAML.

### 3.2. Contratos principais (a implementar)

> Esboço de contratos - a assinatura final é definida na issue `core`. Servem para
> garantir que cada questão (Q1-Q6) seja uma **implementação plugável**.

- **`ModelProvider`** - `load() -> (model, tokenizer)`. Implementações:
  `LocalModelProvider`, `CloudModelProvider`. (Resolve o desacoplamento local<->nuvem.)
- **`Trainer`** (Strategy) - `train(model, dataset, config) -> TrainResult`.
  Implementações: `ContinualPretrainTrainer`, `SupervisedFineTuner`,
  `LoraFineTuner`/`QLoraFineTuner`, `DistillationTrainer`.
- **`Evaluator` / `Metric`** - `evaluate(model, benchmark) -> dict[str, float]`.
  Métricas: `Perplexity`, `CrossEntropy`, `TokenAccuracy`, `QABenchmark`.
- **`DatasetLoader`** - `load() -> Dataset`. Inclui `PdfToTextLoader`,
  `QAPairGenerator` (gera os >=1.000 pares para SFT).
- **`Registry` / `build_from_config`** - mapeia chaves de config -> classes,
  permitindo registrar novas implementações sem editar o núcleo (OCP).
- **`RagPipeline`** - composição `Retriever` + `Generator` + (opcional) `Reflector`.
- **`GuardrailLayer`** - cadeia componível de filtros de entrada/saída.

## 4. Convenções

- **Idioma:** código, identificadores e docstrings em **inglês**; `CONTEXT.md`,
  `PROJECT_CONTEXT.md` e issues em **português**.
- **Config-first:** nada de hiperparâmetros hardcoded; tudo via `configs/`.
- **Reprodutibilidade:** seed global fixada; resultados de benchmark salvos.
- **Git:** push sempre sob a conta do usuário; sem coautoria de IA nos commits
  (ver `CLAUDE.md`).

## 5. Funções complexas

> Seção viva - documentar aqui cada função/algoritmo não trivial à medida que for
> criado.

### Métricas intrínsecas (`evaluation/metrics.py`)
Operam sobre arrays NumPy de `logits` e `labels` já alinhados (o `Evaluator`
aplica o shift causal: prevê o token t+1 a partir dos tokens <= t). Tokens iguais
a `ignore_index` (-100) são ignorados. Em log natural (nats):
- `cross_entropy` = média por token de `-log p(label)`, via log-softmax estável.
- `perplexity` = `exp(cross_entropy)`.
- `token_accuracy` = fração de tokens cujo argmax dos logits é igual ao label.

São acumuladores (`update`/`compute`/`reset`), o que permite alimentar várias
métricas numa única passada de forward.

### Remoção de boilerplate (`data/pdf_loader.py::_drop_repeated_lines`)
Heurística para remover cabeçalhos/rodapés de diários: conta as linhas que se
repetem em pelo menos `boilerplate_threshold` das páginas (mínimo de 2) e as
descarta. Desativada quando há menos de 3 páginas ou `threshold >= 1.0`.

### Empacotamento em blocos (`data/text_corpus.py::chunk_token_ids`)
Para o pré-treino contínuo (Q1), os documentos são tokenizados, concatenados (com
EOS entre eles) e divididos em blocos de `block_size` tokens. O resto final menor
que um bloco é descartado por padrão (`drop_remainder=True`). O agrupamento é uma
função pura, testável sem torch; o `ContinualPretrainTrainer` a usa para montar o
dataset de treino.
