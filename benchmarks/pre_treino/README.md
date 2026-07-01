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

## Conjunto held-out de texto de diário (anticontaminação)

Além das P&R conceituais (que já são held-out por terem sido escritas à mão, fora
do corpus), há um segundo medidor: um conjunto de documentos de diário **disjunto
do treino** (gerado com `scripts/diarios_to_text.py --skip <n_treino>`), em
`data/processed/diarios_heldout.jsonl` (git-ignored, regenerável). Mede a
perplexidade antes/depois em **texto de domínio inédito**: como o modelo não viu
esses documentos no treino, a queda reflete generalização, não memorização.
Configs: `configs/eval_diarios_heldout_antes.yaml` (modelo base) e
`configs/eval_diarios_heldout_depois.yaml` (checkpoint treinado).

Há dois conjuntos de held-out, conforme a escala do treino. O run de 2.000 docs usa
`data/processed/diarios_heldout.jsonl` (docs 2000-2500, 500 docs). O run do córpus
completo (treino em `data/processed/diarios_txt_full`, 68.440 docs) usa
`data/processed/diarios_heldout_full.jsonl` (últimos 2.000 docs, disjunto do treino
completo), com configs `configs/eval_diarios_heldout_full_{antes,depois}.yaml` e
saída em `results/depois_full/`. Os dois held-outs são git-ignored e regeneráveis
com `scripts/diarios_to_text.py` (ver cabeçalho do script).

## Gabarito feito a mão (`benchmarks/a-mao/`)

Há um gabarito de P&R escrito a mão, sem IA (`benchmarks/a-mao/Q1 benchmark P&R.csv`,
separador `;`, colunas `id;instruction;output;arquivo`), como referência independente
do benchmark gerado com auxílio de IA. Cada linha aponta o documento-fonte na coluna
`arquivo` (nome do `.txt` em `data/processed/diarios_txt_full`).

Dois pontos a saber ao usar/auditar este gabarito:

- **Diários repetidos são legítimos.** Várias perguntas podem citar o mesmo `.txt`
  porque cada documento do córpus agrega atos de municípios diferentes na mesma
  página/edição. Logo, mais de uma pergunta por arquivo é esperado, não duplicata
  (ex.: um único arquivo cobre uma dispensa em Paulistana e um contrato em Dom
  Inocêncio). Consequência para a anticontaminação: o held-out é por documento, ciente
  de que um documento mistura municípios.
- **Normalizar a resposta na avaliação.** O texto vem de OCR ruidoso (ex.: `30.000`
  aparece como `30,00o`), então comparação exata de string subconta acertos. Avaliar
  removendo `R$`, separadores de milhar/decimal e tolerando trocas de OCR
  (`o` por `0`).

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
