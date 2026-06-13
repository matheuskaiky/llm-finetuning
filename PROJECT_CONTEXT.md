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
├── .env.example         # (EN) Template de variáveis de ambiente; copiar para .env (ignorado).
├── pyproject.toml       # Dependências + config de ruff/pytest.
├── uv.lock              # Lockfile do ambiente (uv). Versionado.
├── requirements-dev.txt # Subconjunto leve de deps para testes sem o stack de ML.
├── src/llm_finetuning/  # Código-fonte (EN), pacote único. Núcleo + módulos das 6 questões.
├── configs/             # Configuração YAML (modelos/métodos/ambiente). Habilita o OCP.
├── data/                # Datasets brutos e processados. NÃO versionado.
├── models/              # Pesos baixados do HF Hub. NÃO versionado.
├── benchmarks/          # Conjuntos de avaliação. Organizados por fase: pre_treino/ (Q1) e pos_treino/ (Q2/Q3); demais por questão (Q4=100, Q5=30, Q6=30).
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
>   Separados por fase de treino: `pre_treino/` (Q1, diários) e `pos_treino/`
>   (Q2/Q3, docentes, a construir). Dentro de cada fase, a avaliação é antes e
>   depois do treino daquela fase (`results/antes/`, `results/depois/`). A da Q1 é
>   apenas o antes/depois do pré-treino, sem envolver o pós. As tasks seguem o LM
>   Eval Harness (YAML declarativo), conforme `SLIDES_BENCHMARK.md`.
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
├── data/          # [x] DatasetLoader: PdfToTextLoader (PDF->txt), TextCorpusLoader, DocenteExtractor (SIGAA->JSONL, triagem+dedup). (Q&A, splits a seguir.)
├── models/        # [x] ModelProvider: LocalModelProvider, CloudModelProvider (placeholder).
├── evaluation/    # [x] Evaluator + Metrics (perplexidade, entropia, acurácia de token).
├── training/      # [~] Trainers (Strategy): ContinualPretrainTrainer (Q1). SFT/LoRA/distill a seguir.
├── rag/           # [x] GraphRAG (Q5): config, chunking, llm_client, extraction, graph_store (networkx), vector_store (FAISS), retrievers (vector+graph), agent (LangGraph self-reflexivo), pipelines (modos Standard/Agentic como RagRunner + registro), judge, doc_select (deteccao/balanceamento de licitacoes).
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
  `DocenteExtractor` (triagem + extracao de texto + dedup do corpus docente SIGAA
  para JSONL), `QAPairGenerator` (gera os >=1.000 pares para SFT).
- **`Registry` / `build_from_config`** - mapeia chaves de config -> classes,
  permitindo registrar novas implementações sem editar o núcleo (OCP).
- **`RagPipeline`** (Q5, implementado no pacote `rag/`) - `Retriever` (Protocol)
  com `VectorRetriever` (FAISS + bge-m3) e `GraphRetriever` (KG NetworkX);
  `LocalChatLLM` como motor trocável por config (8B bf16 single-GPU, ou 30B FP8
  multi-GPU via `device_map`); agente LangGraph self-reflexivo
  (Analyzer/Router -> Retrieve -> Generate -> Critic, com loop de reflexão). A
  config é `RagConfig` (no próprio pacote, sem tocar no `core`), dirigida por
  `configs/rag_*.yaml`. Scripts: `build_rag_index`, `make_rag_benchmark`, `eval_rag`.
- **`GuardrailLayer`** - cadeia componível de filtros de entrada/saída.

## 4. Convenções

- **Idioma:** código, identificadores e docstrings em **inglês**; `CONTEXT.md`,
  `PROJECT_CONTEXT.md` e issues em **português**.
- **Config-first:** nada de hiperparâmetros hardcoded; tudo via `configs/`.
- **Reprodutibilidade:** seed global fixada; resultados de benchmark salvos.
- **Git:** push sempre sob a conta do usuário; sem coautoria de IA nos commits
  (ver `CLAUDE.md`). **Nunca commitar na `main`:** trabalhar em `dev` ou em
  branches de feature; `main` só recebe via pull request.
- **Segredos:** configuração e segredos em `.env` (ignorado pelo Git); manter o
  `.env.example` versionado em sincronia (ver `CLAUDE.md`, regra 9).

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

### Extracao e dedup do corpus docente (`data/docente_extractor.py`)
Pipeline para o dataset SIGAA (`data/raw/docentesDC-sigaa`, estrutura
`professor/ano/mes/dia/arquivo`). Etapas: (1) triagem por extensao em tres baldes
(`text`/`code`/`noise`); codigo so entra com `include_code=True` (subcorpus opt-in)
e ruido (`.venv`, `__MACOSX`, `_archives/`, binarios) e sempre descartado; (2)
metadados do caminho (professor, ano, mes, dia) e do nome (prefixo numerico de id
do SIGAA); (3) extracao de texto por tipo, com imports tardios (`pypdf` para pdf;
`python-docx` para docx; `python-pptx` para pptx; leitura direta para txt/tex/csv/md;
`html.unescape` + strip para html; doc/ppt legados ficam como `unsupported`); (4)
dedup em duas camadas. A dedup exata (`deduplicate`) agrupa por `text_sha1` (texto
normalizado) quando ha texto, senao por `content_md5`, elege como canonica a versao
mais recente (`max(ano, mes, dia)`, desempate por caminho) e preserva
`duplicated_dates`/`dup_count`; conteudo que aparece em mais de um docente marca
`shared_with_professors` sem atribuir autoria. A dedup aproximada
(`deduplicate_near`, opcional via `near_dedup`) usa MinHash/LSH (`datasketch`) sobre
shingles de palavras, por professor, com limiar de Jaccard configuravel, colapsando
variantes leves. Texto extraido de PDFs e bytes indecodificaveis do filesystem podem
gerar lone surrogates: `sanitize_text` os remove no texto e na linha JSON inteira.
Saida: JSONL, um documento canonico por linha. `export_plaintext_corpus` materializa
o texto canonico (acima de `min_chars`) em `.txt` sob `data/processed`, compativel
com o `TextCorpusLoader` (reaproveita o corpus docente no pre-treino contınuo, Q1).
A logica pura (triagem, metadados, fingerprint, dedup exata e aproximada, export) e
testavel sem o stack de ML.

### Empacotamento em blocos (`data/text_corpus.py::chunk_token_ids`)
Para o pré-treino contínuo (Q1), os documentos são tokenizados, concatenados (com
EOS entre eles) e divididos em blocos de `block_size` tokens. O resto final menor
que um bloco é descartado por padrão (`drop_remainder=True`). O agrupamento é uma
função pura, testável sem torch; o `ContinualPretrainTrainer` a usa para montar o
dataset de treino.

## 6. Testes (TDD)

O fluxo padrão é TDD: escrever primeiro um teste que falha capturando o
comportamento desejado, implementar o mínimo para passar e então refatorar.

- **Rápidos e sem o stack de ML.** A suíte roda só com `requirements-dev.txt`
  (sem torch/transformers/datasets). Por isso as dependências pesadas ficam em
  import tardio e testamos a lógica pura (métricas em numpy, chunking, registry,
  config, normalização de PDF).
- **Testes de contrato genéricos** (`tests/test_contracts.py`): verificam que todo
  componente registrado respeita a interface do núcleo (subclasse correta e método
  exigido presente), sem fixar comportamento específico de cada questão. Novas
  implementações passam a ser checadas automaticamente ao se registrarem.
- **Comandos:** `pytest` e `ruff check .` (config no `pyproject.toml`); rodam
  também na CI e no `pre-commit`.

## 7. Ambiente e assets

**Ambiente Python.** Gerenciado por `uv`, isolado em `.venv` no projeto; nada é
instalado no Python global/conda base. `uv sync` cria o `.venv` a partir do
`uv.lock`; rodar comandos com `uv run ...`. O conda pode ser usado, mas apenas em
env separado, sem sujar o base.

**Hardware de referência.** Máquina de desenvolvimento com 2x NVIDIA L4 (24 GB
cada, ~48 GB no total), que suportam bf16 e FP8. Sem NVLink: modelos maiores que
24 GB precisam ser shardados entre as duas GPUs (tensor-parallel na inferência;
ZeRO/FSDP ou 4-bit no treino). Observação: `nvidia-smi`/NVML falham por mismatch de
driver; o CUDA compute de uma placa funciona normalmente, mas o treino multi-GPU
via NCCL não inicializa (a NCCL chama `nvmlInit` e aborta), então hoje só roda
single-GPU. Detalhe e pedido de suporte: `docs/SUPORTE_INFRA_MULTIGPU.md`.

**Modelo e dataset escolhidos.**

| Asset | ID/Origem | Observação |
|-------|-----------|-----------|
| Base de texto (Q1-Q3) | família `Qwen/Qwen3-*-Base` (densa, `Qwen3ForCausalLM`) | Modelos base (só pré-treino). Escada da Q1 full-parameter: 0.6B e 1.7B (feitos, single-GPU); 4B pronto mas pendente do multi-GPU. Texto puro, vocab 151936. |
| Motor do RAG (Q5) | `Qwen/Qwen3-8B` (instruct, bf16, padrão) e `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (MoE FP8, multi-GPU por `device_map`, reservado p/ quando o NCCL for resolvido). Embeddings `BAAI/bge-m3`. | Trocável por config. O `Qwen3.5-9B` multimodal foi removido (complexo, não cabia na estratégia de texto). |
| Cross-family (Q1/Q5) | `google/gemma-3-1b-pt` e `google/gemma-3-1b-it` (texto puro, gated) | Comparação de família vs Qwen. Gemma 3 4b+ são multimodais. |
| Corpus de diários | `gutoportelaa/dom-pi-corpus-2025` (HF) | Diário Oficial dos Municípios do Piauí 2025; ~195M tokens; parquet, texto na coluna `texto`. Q1/Q5. Variante balanceada (licitações podadas) só para diagnóstico. |
| Dataset docente | Google Drive (zip por grupo) | `docentesDC`/SIGAA. Pasta: drive.google.com/drive/folders/1aDoEszVYDH1-nNoskLSMCfNLN_cjV16K. Estrutura aninhada `TIA-Dados_Professores/grupoN/grupoN.zip`, arquivos por professor (`professor/ano/mes/dia/arquivo`). Q2/Q3/Q5. |

Q1 é full-parameter, limitada por memória: 0.6B cabe numa L4; ~1.7B com FSDP ou
AdamW 8-bit; ~4B no teto (FSDP + activation checkpointing); 9B full-parameter não
cabe nas 2x L4 e fica para LoRA/QLoRA (Q3). Modelos base, não instruct, em Q1-Q3.

**Download dos assets.** IDs vêm do `.env` (`BASE_MODEL_ID`, `DATASET_ID`).
`scripts/download_assets.py` baixa modelo (para `models/`) e dataset (para
`data/raw/`) do HF Hub:

```bash
uv run python scripts/download_assets.py --all
```

O dataset docente vem de uma pasta do Google Drive (zip por grupo). Os links
diretos do Takeout sao presos a sessao do navegador e nao funcionam server-side:
baixar pelo navegador (ou `gdown` se a pasta for "qualquer pessoa com o link") e
colocar os zips em `data/raw/docentesDC-sigaa/`. A extracao achata a camada
`grupoN`, deixando as pastas por professor no topo e os zips originais em
`_archives/`. Ver `README.md`.
