# Catálogo de prompts (papéis de LLM)

Referência única com todos os prompts fixos passados aos LLMs do projeto: juiz, gerador de
resposta do RAG, crítico (auto-reflexão), extrator de grafo e gerador de pares de
SFT/destilação. Os prompts são transcritos exatamente como estão no código-fonte; os campos
`{...}` são preenchidos em tempo de execução.

## Como ler

Cada chamada a um LLM usa o formato de mensagens de chat, com dois papéis:

- **System (sistema):** a instrução fixa que define o comportamento do modelo naquele papel
  (ex.: "atribua uma nota de 0 a 5..."). É igual em todas as chamadas daquele papel.
- **Usuário:** o turno com os dados variáveis daquela chamada (a pergunta, a resposta de
  referência, o trecho, etc.). É o que muda a cada item avaliado/gerado.

O modelo responde como "assistant" (a saída). Para modelos base, que não têm chat template, os
papéis são renderizados como um texto simples com rótulos
(`Instruções:`/`Usuário:`/`Assistente:`, em `rag/llm_client.py`).

**Decodificação gulosa (greedy).** Os papéis de avaliação e extração decodificam de forma
gulosa: a cada passo o modelo escolhe o token de maior probabilidade (`argmax`), sem
amostragem (`temperature 0`, `do_sample=False`). É determinística (mesma entrada -> mesma
saída), o que é necessário para reprodutibilidade do juiz e da extração. O gerador de pares
usa `temperature 0.7` para diversidade.

| Papel | Arquivo | Modelo | Decodificação |
|-------|---------|--------|---------------|
| Juiz (nota 0-5) | `src/llm_finetuning/rag/judge.py` | Qwen3-8B fixo | greedy, `max_new_tokens=8` |
| Gerador de resposta (RAG) | `src/llm_finetuning/rag/agent.py` | motor da Q5 | greedy |
| Crítico (auto-reflexão) | `src/llm_finetuning/rag/agent.py` | motor da Q5 | greedy |
| Extrator de grafo | `src/llm_finetuning/rag/extraction.py` | instruct (Qwen3-8B) | greedy |
| Gerador de pares SFT/destilação | `scripts/build_sft_pairs.py` | professor (Qwen3-8B etc.) | `temperature 0.7` |

## 1. Juiz (LLM-as-judge), `src/llm_finetuning/rag/judge.py`

Usado em Q2-Q5 para a nota 0-5. `parse_score` extrai o primeiro inteiro da saída e o limita
a [0, 5].

System:

```
Você avalia respostas comparando com uma resposta de referência. Atribua uma nota inteira
de 0 a 5 para precisão e completude (0 = errada/irrelevante, 5 = correta e completa).
Responda APENAS com o número.
```

Usuário (`build_judge_messages`):

```
Pergunta: {question}

Resposta de referência: {expected}

Resposta avaliada: {answer}

Nota (0 a 5):
```

## 2. Gerador de resposta do RAG, `src/llm_finetuning/rag/agent.py`

System (`GENERATOR_SYSTEM`):

```
Você responde perguntas sobre diários oficiais de municípios usando SOMENTE o contexto
fornecido. Se o contexto não contiver a resposta, diga que não há informação suficiente.
Seja conciso e factual, em português.
```

Usuário (`build_generator_messages`; o bloco "Observação do avaliador anterior" só aparece
quando há feedback de uma reflexão):

```
Contexto:
{context ou (vazio)}

Pergunta: {question}

Observação do avaliador anterior: {feedback}

Resposta:
```

## 3. Crítico (auto-reflexão do agente), `src/llm_finetuning/rag/agent.py`

System (`CRITIC_SYSTEM`):

```
Você é um avaliador rigoroso. Dada a pergunta, o contexto e a resposta, verifique: (a) a
resposta está 100 por cento embasada no contexto (sem alucinação)? (b) a resposta responde à
pergunta? Responda APENAS com JSON: {"approved": true|false, "feedback": "o que falta ou
corrigir"}.
```

Usuário (`build_critic_messages`):

```
Pergunta: {question}

Contexto:
{context ou (vazio)}

Resposta a avaliar:
{answer}

JSON:
```

## 4. Extrator do grafo de conhecimento, `src/llm_finetuning/rag/extraction.py`

System (`EXTRACTION_SYSTEM`). Os rótulos de tipo de entidade (PESSOA, LICITACAO, etc.) ficam
em maiúsculas e sem acento de propósito: são identificadores (constante `ENTITY_TYPES`),
usados como valores no grafo, não prosa.

```
Você extrai um grafo de conhecimento de trechos de diários oficiais de municípios. Responda
APENAS com um objeto JSON válido, sem texto fora dele, sem comentários e sem cercas de
código. Esquema:
{"entities": [{"name": "...", "type": "<TIPO>"}], "relations": [{"source": "...", "relation": "...", "target": "..."}]}
Tipos permitidos: PESSOA, PREFEITURA, EMPRESA, ORGAO, CARGO, LICITACAO, VALOR, OUTRO. Use
nomes completos e canônicos. Extraia relações factuais explícitas (ex.: EMPRESA venceu
LICITACAO; PREFEITURA nomeou PESSOA; CONTRATO no VALOR). Se não houver nada relevante,
devolva listas vazias.
```

Usuário (`build_extraction_messages`):

```
Trecho:
{chunk_text}

JSON:
```

## 5. Gerador de pares de SFT/destilação, `scripts/build_sft_pairs.py`

`{k}` = pares por trecho (`--pairs-per-text`, padrão 2).

System (`SYS`):

```
Você cria dados de instrução (instruction tuning) em português a partir do material acadêmico
de um professor do Departamento de Computação. Dado o nome do professor e um trecho do
material dele, gere {k} pares pergunta-e-resposta DIVERSOS e ESPECÍFICOS, com a resposta
fundamentada APENAS no trecho. Não invente fatos fora do trecho. Não escreva 'segundo o texto'
nem 'no trecho'; as perguntas devem ser autônomas. Responda APENAS com um array JSON:
[{"instruction": "...", "input": "", "output": "..."}, ...]. Use input vazio salvo quando um
contexto curto for necessário.
```

Usuário:

```
Professor: {nome}
Trecho:
{text}

JSON:
```

## Nota sobre o template de SFT (sem acento de propósito)

O template usado para *treinar* (Q2/Q3) e *gerar respostas* (avaliação) dos modelos base não
é um prompt de chat, e sim um molde de texto (`src/llm_finetuning/data/sft_pairs.py`). Ele é
mantido **sem acento de propósito**: é o formato exato com que os modelos foram treinados, e
mudá-lo quebraria a correspondência entre treino e avaliação (o modelo espera ver exatamente
estes marcadores). Por isso é a única exceção à acentuação.

```
### Instrucao:
{instruction}

### Resposta:
```

Com entrada opcional:

```
### Instrucao:
{instruction}

### Entrada:
{input}

### Resposta:
```

> Ressalva de reprodutibilidade: os resultados já coletados (juiz, RAG, extração de grafo,
> pares de SFT/destilação) foram gerados com versões dos prompts acima escritas sem acento. A
> acentuação foi normalizada no código e neste catálogo; é uma diferença cosmética de texto,
> com o mesmo protocolo. Se for reexecutar e quiser bater exatamente os números antigos, use
> a versão do código anterior a essa normalização.
