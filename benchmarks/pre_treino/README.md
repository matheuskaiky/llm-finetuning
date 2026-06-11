# Benchmark de pré-treino (Q1 - diários)

Benchmark de domínio da fase de **pré-treino** (Questão 1: pré-treino contínuo
sobre o corpus de diários oficiais dos municípios do Piauí). Construído seguindo a
metodologia de `SLIDES_BENCHMARK.md` (LM Eval Harness, definição declarativa em
YAML, métricas por formato de tarefa).

Organização geral dos benchmarks por fase de treino:

- `benchmarks/pre_treino/` - esta pasta, fase de pré-treino (Q1).
- `benchmarks/pos_treino/` - fase de pós-treino (Q2 SFT e Q3 LoRA/QLoRA), sobre o
  docentesDC; reservada, a construir depois. Não faz parte da avaliação da Q1.

## Objetivo

Medir a qualidade do LLM no domínio dos diários **antes e depois** do pré-treino
contínuo. A avaliação da Q1 é apenas esse antes/depois do pré-treino; não envolve a
fase de pós-treino. A tarefa pede como métricas a perplexidade, a entropia cruzada
e a acurácia de previsão de tokens; o conjunto traz pelo menos 25 perguntas com
respostas de referência sobre atos e conceitos do domínio.

## Antes e depois (do pré-treino)

A mesma bateria é avaliada em dois momentos. Os resultados ficam separados:

- `results/antes/` - modelo **antes** do pré-treino contínuo (linha de base).
- `results/depois/` - modelo **depois** do pré-treino contínuo.

A comparação antes vs depois é a evidência central da Q1: espera-se queda da
perplexidade e da entropia cruzada e ganho de acurácia de token após a adaptação
ao domínio.

## Formatos e métricas (conforme SLIDES_BENCHMARK.md)

O conjunto de referência (`diarios_qa.jsonl`, um objeto por linha com
`instruction` e `output`) alimenta duas tasks do LM Eval Harness:

- `lm_eval/diarios_qa_ppl.yaml` - métricas **intrínsecas** (`loglikelihood_rolling`):
  perplexidade por palavra e por byte, e bits por byte (proxy da entropia
  cruzada). É a task alinhada às métricas exigidas pela Q1.
- `lm_eval/diarios_qa_gen.yaml` - **QA factual** em `generate_until` com **BLEU**
  contra a resposta de referência, no estilo da tarefa de QA factual dos slides.
  Decodificação gulosa (determinística).

A tarefa de classificação/roteamento descrita nos slides (múltipla escolha) não se
aplica ao domínio dos diários e foi deixada de fora, por delimitação.

## Requisitos de qualidade das questões (slides, seção 4)

Relevância, avaliabilidade, realismo, delimitação e análise de erro. As perguntas
são factuais e conceituais sobre o domínio (atos administrativos, licitações,
orçamento, regime de pessoal), com respostas curtas e verificáveis, evitando
afirmações específicas não confirmáveis.

## Como rodar

Com o LM Eval Harness instalado (`lm-eval`), a partir da raiz do repositório:

```bash
# Antes do pré-treino: usar o modelo base
lm_eval --model hf --model_args pretrained=<modelo> \
  --tasks diarios_qa_ppl,diarios_qa_gen \
  --include_path benchmarks/pre_treino/lm_eval \
  --output_path benchmarks/pre_treino/results/antes

# Depois do pré-treino: apontar para o checkpoint treinado e gravar em results/depois
```

As mesmas métricas intrínsecas também podem ser computadas pelo avaliador interno
do projeto: `configs/eval_diarios_antes.yaml` e `configs/eval_diarios_depois.yaml`.

## Arquivos

- `diarios_qa.jsonl` - conjunto de referência (>= 25 pares).
- `lm_eval/diarios_qa_ppl.yaml` - task de perplexidade/entropia.
- `lm_eval/diarios_qa_gen.yaml` - task de QA factual (BLEU).
- `results/antes/`, `results/depois/` - saídas de avaliação (geradas; JSON não
  versionado).
