# IMPLEMENTACAO_QUESTOES.md - Como cada questao e feita

> Mapa pratico das seis questoes (Q1-Q6): o que cada uma pede, quais modulos,
> configs, scripts, benchmarks e arquivos de resultado entram, como esta
> implementada e as decisoes tomadas. Complementa o
> [`PROJECT_CONTEXT.md`](../PROJECT_CONTEXT.md) (arquitetura/contratos) e o
> [`CONTEXT.md`](../CONTEXT.md) (produto/cronograma).

## Visao geral do fluxo

Tudo e dirigido por configuracao (OCP): um experimento e uma config YAML em
`configs/` resolvida pelos registries do nucleo (`core/registry.py`). Os scripts
em `scripts/` sao invólucros finos de CLI; o codigo de produção vive em
`src/llm_finetuning/`; os jobs de cluster ficam em `scripts/slurm/`. Cada
avaliacao escreve um CSV versionado em `results/` (mais um `*_details.jsonl`
git-ignored com as respostas cruas quando ha juiz).

| Camada | Onde |
|--------|------|
| Logica de treino/avaliacao | `src/llm_finetuning/` |
| O que rodar (modelo, metodo, hiperparametros) | `configs/*.yaml` |
| Entrypoints CLI | `scripts/*.py` |
| Jobs de cluster (SLURM) | `scripts/slurm/*.sbatch` |
| Conjuntos de avaliacao | `benchmarks/` |
| Resultados versionados | `results/*.csv` + `results/README.md` |

**Modelos cobertos (transversal a Q1-Q4).** Escada Qwen3 base (0.6B, 1.7B; 4B
bloqueado por hardware), segunda familia `gpt2` (124M/355M/774M) e cross-family
`gemma-3-1b` (pt/it). A motivacao da segunda familia e mostrar que o pipeline e
agnostico de arquitetura e vocabulario (gpt2 tem vocab BPE ingles de 50257, contra
151936 do Qwen).

**Quantidade e origem das perguntas por questao.** "Derivada do dataset" = gerada por
LLM ancorada em trechos reais do corpus (ou no grafo extraido), amostragem com seed 42.

| Questao | Conjunto | Qtd. | Origem |
|---------|----------|------|--------|
| Q1 | benchmark de P&R (`diarios_qa.jsonl`) | 33 | conceituais sobre o dominio, held-out por construcao (auxilio de IA) |
| Q2 | pares de SFT (treino) | 1.000 | derivadas do docentesDC (LLM ancorado no trecho) |
| Q2/Q3 | held-out / recall | 150 / 150 | derivadas dos textos-fonte (recall exclui as perguntas de treino) |
| Q3 | LoRA | reusa Q2 | mesmo conjunto da Q2 (so muda o metodo) |
| Q4 | pares de destilacao (treino) | 1.200 | derivadas dos diarios (professor ancorado no trecho); 400/professor na ablacao |
| Q4 | benchmark recall / held-out | 100 / 100 | derivadas dos diarios |
| Q5 | benchmark RAG (`diarios_rag_30.jsonl`) | 30 | derivadas do indice: ~60% factual de chunk, ~40% multi-hop do grafo (+ variantes balanced/naolic) |
| Q6 | guardrails (`guardrails_30.jsonl`) + parafraseados | 30 + 15 | prompts de seguranca construidos (nao vem do corpus) |

---

## Q1 - Pre-treino continuo

**Pedido.** Continuar o pre-treino de um LLM base sobre os diarios oficiais e
medir antes/depois (perplexidade, entropia cruzada, acuracia de previsao de token)
num benchmark de pelo menos 25 perguntas.

**Implementacao.**
- Trainer: `training/pretrain.py::ContinualPretrainTrainer` (registrado como
  `continual_pretrain`). Empacota o corpus em blocos de `block_size` tokens
  (`data/text_corpus.py::chunk_token_ids`, funcao pura) e roda o `Trainer` da HF
  com collator causal.
- Avaliacao: `evaluation/evaluator.py` + `evaluation/metrics.py` (perplexidade,
  entropia, acuracia de token; acumuladores que rodam numa unica passada).
- Scripts: `scripts/train.py` (treino), `scripts/evaluate.py` (metricas
  intrinsecas), `scripts/eval_q1_gpt2.py` (escada gpt2: held-out / qa / OOD).
- Configs: `pretrain_diarios_qwen3_0p6b.yaml`, `..._1p7b.yaml`, `..._4b.yaml`,
  `pretrain_diarios_gpt2{,_medium,_large}.yaml`, `pretrain_diarios_gemma3_1b{,_it}.yaml`.
- Benchmark: `benchmarks/pre_treino/diarios_qa.jsonl`.
- Resultados: `results/q1_base_vs_instruct.csv`, `results/q1_gpt2.csv`,
  `results/q1_forgetting.csv`, `results/q1_balanceamento_licitacao.csv`.

**Decisoes.**
- **Modelos base, nao instruct.** Para que o antes/depois isole o efeito do nosso
  treino, e nao o alinhamento de fabrica de um instruct.
- **LR baixo.** Pre-treino continuo usa LR pequeno para nao destruir o
  conhecimento previo (esquecimento catastrofico).
- **Escada de tamanho limitada por hardware.** 0.6B e 1.7B cabem full fine-tune em
  uma L4; 4B nao cabe (FSDP multi-GPU nao inicializa por NVML quebrada). O 4B fica
  documentado como limite, nao como resultado.
- **Remocao do 4B/8B dos resultados.** Sem antes/depois valido, foram retirados do
  CSV para nao poluir a comparacao.
- **Diagnostico de esquecimento.** `scripts/eval_forgetting.py` mede a
  perplexidade OOD (sonda `docentesDC`) antes/depois; `q1_forgetting.csv` guarda o
  delta. Qwen pequenos esquecem um pouco (delta positivo); gpt2 nao (delta
  negativo, pois parte de uma base muito ruim em PT).
- **Ablacao de licitacoes.** O corpus tem excesso de editais de licitacao
  repetitivos; `q1_balanceamento_licitacao.csv` compara o corpus cheio vs a
  variante podada (`scripts/build_balanced_corpus.py`) so para diagnostico.

---

## Q2 - Pos-treino supervisionado (SFT)

**Pedido.** Gerar pelo menos 1.000 pares `{instruction, input?, output}` do
`docentesDC` e fazer SFT; avaliar antes/depois.

**Implementacao.**
- Geracao de pares: `data/sft_pairs.py` (template, `build_prompt`,
  `build_input_and_labels` com mascara de loss so na resposta) +
  `scripts/build_sft_pairs.py` (gera os pares via um LLM motor; aceita
  `--device-map` e `--load-in-4bit`).
- Trainer: `training/sft.py::SupervisedFineTuneTrainer` (SFT completion-only:
  tokeniza prompt e resposta separadamente e mascara o prompt com -100).
- Avaliacao: `scripts/eval_sft.py` (juiz LLM 0-5 sobre um conjunto de recall).
- Configs: `sft_docentes_qwen3_0p6b.yaml`, `..._1p7b.yaml`, `..._4b.yaml`,
  `sft_docentes_gpt2{,_medium,_large}.yaml`, `sft_docentes_gemma3_1b.yaml`.
- Resultados: `results/q2_sft.csv`, `results/q2_data_curve.csv`,
  `results/q2_sft_eval_*_recall.csv`.

**Decisoes.**
- **SFT do modelo ja pre-treinado na Q1.** O student da Q2 parte do checkpoint da
  Q1 (ex.: gpt-q1-fft), encadeando pre-treino continuo e SFT.
- **Loss so na resposta.** Mascarar o prompt evita treinar o modelo a reproduzir a
  pergunta; concentra o sinal na geracao desejada.
- **Juiz LLM 0-5.** Como nao ha gabarito unico, a qualidade da resposta e medida
  por um juiz LLM fixo (Qwen3-8B) com nota 0-5, garantindo comparabilidade entre
  modelos.
- **gpt2 incluido para contraste.** A familia gpt2 melhora a perplexidade
  (adapta a lingua) mas o juiz fica <= 0.5: ela aprende o formato, nao a tarefa em
  portugues. E um resultado negativo informativo, mantido de proposito.
- **Curva de dados.** `q2_data_curve.csv` registra o juiz em funcao do numero de
  pares, para justificar o tamanho do conjunto de SFT.

---

## Q3 - Pos-treino com PEFT (LoRA/QLoRA)

**Pedido.** Repetir o experimento da Q2 com LoRA e/ou QLoRA e comparar com o SFT
pleno (qualidade e custo).

**Implementacao.**
- Reusa `training/sft.py`: quando a config traz um bloco `peft`,
  `build_lora_kwargs` traduz para `LoraConfig` e o mesmo trainer aplica LoRA/QLoRA
  por extensao (nao ha trainer separado).
- Targets de LoRA: `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`
  (Qwen3/gemma). Para gpt2 a config sobrescreve com `c_attn/c_proj/c_fc`.
- Sweeps: `scripts/q23_sweeps.py` (varredura de rank).
- Configs: `lora_docentes_qwen3_0p6b.yaml`, `..._1p7b.yaml`,
  `lora_docentes_gpt2{,_medium,_large}.yaml`, `lora_docentes_gemma3_1b.yaml`.
- Resultados: `results/q3_lora.csv`, `results/q3_rank_sweep.csv`,
  `results/q3_lora_*_recall.csv`.

**Decisoes.**
- **Um unico trainer, dois metodos.** SFT pleno e LoRA compartilham o
  `SupervisedFineTuneTrainer`; o que muda e a config. Isso e a aplicacao direta do
  OCP e mantem a comparacao justa (mesmo loop, mesmos dados).
- **Targets por familia.** gpt2 usa nomes de modulo diferentes (`c_attn` etc.);
  errar isso fez jobs falharem com "target_modules not found", corrigido via config
  e nao via codigo.
- **Varredura de rank.** `q3_rank_sweep.csv` mede o trade-off qualidade x custo do
  rank de LoRA, base da comparacao com o SFT pleno pedida pela questao.

---

## Q4 - Destilacao de conhecimento

**Pedido.** Definir teacher e student, destilar via dataset sintetico, avaliar num
benchmark de 100 perguntas e analisar se houve transferencia de conhecimento.

**Implementacao (duas vias).**
- **Response-based (principal).** O professor gera Q&A sintetico
  (`scripts/build_sft_pairs.py`) e o student e treinado por SFT nesse conjunto
  (`training/sft.py`). Nao exige vocabulario compartilhado, entao permite cruzar
  familias (ex.: professor gemma -> student qwen).
- **Logit-KD (comparacao).** `training/distill.py::DistillationTrainer` e
  `kd_loss` (KL(teacher||student) temperada + CE, so nos tokens de resposta).
  Exige mesmo tokenizer/vocab (mesma familia).
- Scripts: `scripts/build_sft_pairs.py` (datagen), `scripts/train.py` (com
  `--data-path` para o conjunto por professor), avaliacao via juiz fixo.
- Configs: `distill_diarios_response.yaml`, `distill_logitkd_qwen3_0p6b.yaml`,
  `distill_diarios_gpt2.yaml`.
- Jobs: `q4_datagen_teacher.sbatch` (1 GPU), `q4_datagen_30b.sbatch` (2 GPUs),
  `q4_distill_teacher_students.sbatch` (loop de 7 students por professor),
  `q4_eval_teacher.sbatch`.
- Resultados: `results/q4_distill.csv`, `results/q4_methods.csv`,
  `results/q4_teacher_compare.csv`, `results/q4_teacher_t{8b,30b,27b,31b}_recall.csv`.

**Decisoes.**
- **Quatro professores comparados.** Qwen3-8B (baseline), gemma-3-27b-it,
  gemma-4-31b-it e Qwen3-30B foram usados com todos os students; a media do juiz
  diz quem foi melhor professor. Resultado: Qwen3-30B na frente (0.354), mas a
  margem entre professores e pequena (~0.04).
- **O student domina o professor.** A escolha de student importa mais que a de
  professor: um student pequeno bem destilado supera professores maiores na tarefa
  especifica. Isso motivou reaproveitar os students na Q5.
- **Juiz fixo (Qwen3-8B, cuda:1).** Para comparar professores e students entre si,
  o juiz e sempre o mesmo, e nao auto-avaliacao.
- **Transfer ratio.** Alem do juiz, mede-se a razao de transferencia
  (queda de perplexidade da resposta) para distinguir "aprendeu lingua" de
  "aprendeu a tarefa". gpt2 mostra ppl 1537->134 mas juiz 0.05: adaptou forma, nao
  conteudo.
- **Pipeline encadeado em SLURM.** datagen -> distill -> eval encadeados por
  `afterok`, rodando os 4 professores x 7 students de forma autonoma.

---

## Q5 - RAG (GraphRAG self-reflexivo)

**Pedido.** Aplicacao RAG (Standard, Agentic ou Self-Reflective) sobre os datasets,
benchmark de 30 perguntas, medindo a contribuicao do RAG.

**Implementacao.** Pacote `rag/` com config propria (`RagConfig`, sem tocar no
`core`):
- Indexacao: `chunking.py`, `vector_store.py` (FAISS + `BAAI/bge-m3`),
  `extraction.py` + `graph_store.py` (grafo de conhecimento em NetworkX).
- Recuperacao: `retrievers.py` (`VectorRetriever`, `GraphRetriever`), `rerank.py`.
- Motor: `llm_client.py::LocalChatLLM` (trocavel por config; 4-bit NF4 opcional;
  `device_map` aceita `str` ou `dict` para pinar GPU).
- Modos: `pipelines.py` (`StandardRunner`, `AgenticRunner` registrados como
  `RagRunner`) e `agent.py` (agente LangGraph Analyzer/Router -> Retrieve ->
  Generate -> Critic, com loop de reflexao).
- Avaliacao: `judge.py` + `scripts/eval_rag.py` (com `--llm-model` para reusar uma
  config em varios motores e `--judge-model`/`--judge-device` para o juiz fixo) e
  `scripts/eval_retrieval.py` (hit-rate@k do retriever).
- Licitacoes: `doc_select.py` (deteccao/balanceamento) e `RAG_ROADMAP.md`.
- Configs: `rag_diarios.yaml`, `rag_diarios_qwen3_30b.yaml`,
  `rag_diarios_gemma27b.yaml`, `rag_diarios_gemma31b.yaml`,
  `rag_diarios_gemma3_1b_{pt,it}.yaml`, variantes balanced/dedup/naolic.
- Benchmark: `benchmarks/rag/diarios_rag_30.jsonl` (+ variantes balanced/naolic).
- Resultados: `results/q5_engines.csv` (leaderboard de motores x modo),
  `results/q5_retrieval.csv`, `results/q5_student_*.csv`,
  `results/q5_engine_gemma-*.csv`, `results/q5_qualitativos.md`.

**Decisoes.**
- **Students da Q4 como motores de RAG.** Ideia central desta fase: testar os
  modelos destilados como motor leve. O qwen2.5-0.5b-distill salta de 0.07
  (baseline, sem contexto) para 3.87 com RAG standard, superando o Qwen3-8B (2.70).
  Mostra que destilacao especializada + recuperacao bate um modelo geral 16x maior.
- **Professores grandes como motor.** gemma-3-27b-it (4-bit) roda limpo pinado em
  uma L4 e atinge 3.10 (acima do 8B). gemma-4-31b-it (4-bit) nao cabe ao lado do
  juiz numa L4 (OOM em algumas perguntas): documentado como limite de hardware, com
  o numero parcial preservado.
- **Juiz fixo no cuda:1.** O motor de RAG ocupa o cuda:0; o juiz Qwen3-8B fica
  fixo no cuda:1 para comparabilidade entre motores. Espalhar o motor 4-bit pelas
  duas GPUs causou OOM no juiz (todas as respostas viraram `<error>`, juiz 0.00); a
  correcao foi pinar o motor com `device_map {"": 0}`.
- **Contribuicao do RAG = baseline vs com recuperacao.** Cada motor e medido sem
  contexto (baseline) e com os modos standard/agentic/agentic_graph; a diferenca e
  a contribuicao pedida pela questao.
- **Notebook CSV-driven.** O grafico de motores le `q5_engines.csv` direto; nova
  linha no CSV atualiza o leaderboard sem editar codigo.

---

## Q6 - Guardrails

**Pedido.** Camada de protecao (bloqueio/reescrita/classificacao/mascaramento),
benchmark de 30 perguntas, medindo o grau de protecao adicionado.

**Implementacao.** Pacote `guardrails/`:
- `core.py`: `GuardrailResult`, a base `Guardrail` (estagios `input`/`output`) e a
  `GuardrailLayer` que encadeia filtros (bloqueio interrompe a cadeia;
  mascaras/reescritas acumulam).
- `filters.py`: `PiiMaskGuardrail` (mascaramento de PII), `JailbreakGuardrail`
  (bloqueio de jailbreak), `UnsafeTopicGuardrail` (classificacao de topico).
- `pii.py`: deteccao de dados pessoais.
- Avaliacao: `scripts/eval_guardrails.py`.
- Benchmarks: `benchmarks/guardrails/guardrails_30.jsonl`,
  `guardrails_adversarial.jsonl`.
- Resultados: `results/q6_guardrails.csv`, `results/q6_adversarial.csv`.

**Decisoes.**
- **Estrategias componiveis (OCP).** Cada protecao e um `Guardrail`; a camada e uma
  lista. Adicionar uma protecao e adicionar uma classe, sem mexer na camada.
- **Dois estagios.** Filtros agem na entrada (bloquear prompt malicioso) e/ou na
  saida (mascarar PII gerada), conforme `stages`.
- **Metrica de duas faces.** Mede-se a taxa de bloqueio de conteudo nocivo e a
  preservacao de respostas legitimas (nao basta bloquear tudo). O conjunto
  adversarial (`q6_adversarial.csv`) testa robustez a tentativas de contorno.

---

## Avaliacao e reprodutibilidade (transversal)

- **Juiz LLM fixo.** Q2/Q3/Q4/Q5 usam o mesmo juiz (Qwen3-8B) com nota 0-5 para
  comparabilidade; as respostas cruas ficam nos `*_details.jsonl` (git-ignored).
- **Metricas por questao.** Q1: perplexidade/entropia/acuracia de token +
  delta de esquecimento OOD. Q2/Q3: juiz + perplexidade da resposta. Q4: juiz +
  transfer ratio. Q5: juiz por modo + hit-rate@k do retriever. Q6: taxa de
  protecao + preservacao.
- **CSV versionado.** Todo numero do relatorio e do notebook vem de `results/*.csv`;
  o `results/README.md` descreve cada arquivo.
- **Notebook.** `notebooks/graficos_resultados.ipynb` e totalmente CSV-driven (um
  painel por questao); regenerar um CSV atualiza os graficos.
- **Configs e seeds.** Cada resultado aponta para a config que o gerou e a seed
  global e fixa (`set_global_seed`).
