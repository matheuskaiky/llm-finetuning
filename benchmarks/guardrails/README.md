# Benchmark de guardrails (Q6 - camada de segurança)

Conjunto para medir o grau de proteção adicionado pela camada de guardrails sobre o
assistente. A avaliação compara o comportamento **sem** e **com** a camada: quanto
das entradas nocivas é bloqueado/mascarado e quanto das entradas legítimas é
preservado (sem falsos positivos).

## Arquivos

- `guardrails_30.jsonl` - 30 itens rotulados, um objeto por linha
  (`{"id", "text", "type"}`). Composição:
  - `jailbreak` (5) - tentativas de burlar instruções ("ignore as instruções...").
  - `unsafe` (5) - pedidos de conteúdo perigoso/ilícito.
  - `pii_output` (5) - casos em que a resposta tenderia a expor dados pessoais (PII).
  - `benign` (15) - perguntas legítimas, usadas para medir falsos positivos.
- `guardrails_adversarial.jsonl` - 15 prompts adversariais extras (reforço de
  jailbreak/unsafe), para estressar os filtros além do conjunto principal.

## Camada avaliada (`src/llm_finetuning/guardrails/`)

A camada é uma cadeia componível (`GuardrailLayer`) de filtros de entrada e saída:

- `JailbreakGuardrail` - bloqueia tentativas de burla de instrução.
- `UnsafeTopicGuardrail` - classifica e barra temas inseguros.
- `PiiMaskGuardrail` - mascara PII na saída.

Cada guardrail é uma classe registrada (OCP): um novo filtro entra por extensão, sem
mudar a camada nem o avaliador.

## Métrica

`scripts/eval_guardrails.py` roda o benchmark com e sem a camada e reporta, por
`type`, a fração de entradas nocivas neutralizadas (bloqueio/mascaramento) e a
fração de entradas benignas preservadas (taxa de falso positivo). O grau de proteção
é o salto entre os dois cenários.

## Como rodar

```bash
python scripts/eval_guardrails.py \
  --benchmark benchmarks/guardrails/guardrails_30.jsonl \
  --out results/q6_guardrails.csv
```

Resultado consolidado em `results/q6_guardrails.csv` e a síntese em
[`results/README.md`](../../results/README.md): a camada leva o bloqueio/mascaramento
de 0 para 100% nas categorias nocivas, sem falsos positivos nos itens benignos.
