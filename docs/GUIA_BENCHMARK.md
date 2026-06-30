# Guia: como construir um benchmark de qualidade

Documento de método. Define o que separa um benchmark que mede de um que apenas
enche linha, e dá regras concretas para os quatro conjuntos de perguntas do projeto:
Q1 (P&R sobre os diários), Q4 (recall da destilação), Q5 (RAG) e Q6 (guardrails). As
quantidades e os arquivos de cada um estão em [`PERGUNTAS.md`](PERGUNTAS.md).

O princípio que orienta tudo: **uma pergunta só vale se a resposta certa depende da
capacidade que você quer medir.** Se um modelo sem treino, sem recuperação e sem
contexto já acerta, a pergunta não mede nada do que o projeto faz.

## 1. O problema das perguntas rasas

A falha mais comum é a pergunta enciclopédica/definicional, do tipo "O que é X?". Ela
parece razoável, mas é rasa por três motivos:

1. **Não discrimina.** A definição de "portaria" ou de "licitação" está no
   pré-treino de qualquer LLM de uso geral. O modelo base já responde igual ao
   modelo adaptado ao domínio, então a métrica antes/depois não se move. Mede o
   conhecimento de fábrica, não o efeito do nosso treino/recuperação.
2. **Não é ancorada nos dados.** Não exige nada que esteja especificamente no corpus
   dos diários do Piauí. Poderia ser respondida sobre qualquer município do Brasil.
3. **É difícil de avaliar objetivamente.** "Explique o que é um decreto" admite mil
   redações certas; vira nota subjetiva de juiz, não acerto verificável.

Exemplo real, do nosso próprio `benchmarks/pre_treino/diarios_qa.jsonl`:

```
RUIM:  "O que é uma portaria?"
RUIM:  "O que é o Diário Oficial do Município?"
```

Essas perguntas existem no benchmark da Q1 e são justamente as que menos contribuem
para a comparação base vs pré-treinado. Reescritas para medir adaptação ao domínio:

```
MELHOR: "Em um decreto de abertura de credito suplementar publicado no diario,
         qual rubrica orcamentaria e citada como fonte do recurso?"
MELHOR: "Numa portaria de nomeacao tipica dos diarios, que tres campos identificam
         o servidor nomeado e o cargo?"
```

A versão melhor exige ter visto a *forma* dos atos do corpus, não o conceito
genérico. Regra prática: se trocar "Piauí/diário/prefeitura" por outro lugar
qualquer e a pergunta continuar idêntica, ela é rasa.

## 2. Princípios gerais (valem para os quatro benchmarks)

- **Relevância.** A pergunta exercita a capacidade-alvo da questão (adaptação de
  domínio na Q1, transferência na Q4, recuperação na Q5, segurança na Q6). Nada de
  trivia fora do escopo.
- **Avaliabilidade.** A resposta de referência é curta, específica e verificável
  (um número, um nome, uma data, um artigo, um valor). Fuja de respostas-redação:
  elas inflam o ruído do juiz.
- **Ancoragem na fonte.** A resposta certa tem que estar no corpus/contexto que o
  sistema usa. Se está só no conhecimento geral do LLM, não testa o pipeline.
- **Dificuldade calibrada e discriminação.** Um bom conjunto separa sistemas: o fraco
  erra, o forte acerta. Perguntas que todos acertam (rasas) ou que todos erram
  (impossíveis/ambíguas) não discriminam. Misture níveis.
- **Anticontaminação (held-out).** As perguntas e os textos que as embasam não podem
  estar no conjunto de treino. Caso contrário você mede memorização, não
  generalização. Na Q1 o held-out de diário é gerado disjunto do treino
  (`diarios_to_text.py --skip <n_treino>`); na Q2/Q3 o recall exclui as perguntas de
  treino.
- **Delimitação e não-ambiguidade.** Uma única interpretação e uma única resposta
  certa. Se dois leitores humanos discordam do gabarito, a pergunta é ruim.
- **Cobertura balanceada.** Distribua entre os tipos de ato/categoria reais; não
  encha o conjunto com 80% de licitação só porque é o que mais aparece.
- **Independência de formato.** Não dê pistas da resposta na pergunta nem padronize
  de um jeito que o modelo decora ("a resposta sempre é a primeira data").

## 3. Anti-padrões (e a correção)

| Anti-padrão | Por que é ruim | Correção |
|-------------|----------------|----------|
| "O que é X?" (definicional) | Resposta no pré-treino geral; não discrimina | Pergunte um fato concreto do corpus |
| Resposta longa/dissertativa | Vira nota subjetiva, alta variância do juiz | Resposta curta e única |
| Resposta na própria pergunta | Mede cópia, não conhecimento | Remova a pista |
| Sim/Não | 50% de acerto no chute | Peça o dado que justifica |
| Pergunta ambígua ou com gabarito discutível | Ruído; juízes humanos discordam | Reescreva até resposta única |
| Só o tipo de ato mais frequente | Não cobre o domínio; infla um padrão | Balanceie por categoria |
| Gabarito ruidoso (texto cru truncado) | "Acerto" fica indefinido | Normalize o gabarito |

Sobre o último: no `benchmarks/rag/diarios_rag_30.jsonl` há `expected_answer` como
`"1.719.0"` e `"5.12.2"`, fragmentos truncados de número de ato/artigo. Mesmo numa
pergunta boa, um gabarito mal normalizado torna o acerto ambíguo. Padronize o
formato da resposta (sem reticências, sem corte no meio de um número).

## 4. Guia por benchmark

### Q1 - P&R sobre os diários (`benchmarks/pre_treino/diarios_qa.jsonl`)

Objetivo: medir o efeito do **pré-treino contínuo** no domínio (antes vs depois).

- A pergunta deve depender da *forma e do conteúdo* dos atos do corpus, não da
  definição genérica do instituto jurídico. Esse é o ponto que separa Q1 de um
  quiz de direito administrativo.
- Resposta curta e factual. Como a métrica principal é intrínseca (perplexidade,
  entropia, acurácia de token sobre texto de domínio), o conjunto de P&R serve de
  sonda complementar: priorize perguntas cuja resposta certa o modelo só produz se
  internalizou o estilo dos diários.
- Cubra as classes de ato reais: decreto, portaria, edital de licitação, nomeação,
  orçamento/crédito suplementar, regime de pessoal. Evite concentrar em uma só.
- Held-out por construção: as P&R são escritas à mão, fora do corpus de treino.
- Acompanhe sempre o held-out de **texto de diário inédito** (150 docs disjuntos):
  é ele que mede generalização sem depender do juiz.

### Q4 - recall da destilação (`data/processed/distill/diarios_distill_recall.jsonl`)

Objetivo: medir se o **conhecimento do professor passou para o aluno** (100
perguntas de recall in-domain).

- Cada pergunta deve ter resposta **única, curta e ancorada num trecho específico**
  do diário usado para gerar o par (área do imóvel, nome do donatário, número do
  processo, prazo do contrato). Bons exemplos do conjunto atual:
  "Qual é a área do imóvel doado?" -> "50,1871 ha".
- Como o conjunto é gerado por um LLM professor, há risco de **circularidade** (o
  mesmo tipo de modelo gera e depois é avaliado). Mitigue: revise uma amostra à mão,
  descarte perguntas cujo gabarito não esteja literalmente no texto-fonte, e
  remova respostas que o professor "alucinou".
- Cuidado com o modo de falha de geração: modelos pequenos repetem e despejam pares
  extras na resposta (visto nos `*_recall_details.jsonl`). Isso é problema do
  *modelo*, não da pergunta, mas o benchmark deve ter gabarito curto o bastante para
  o juiz pontuar só o trecho relevante.
- Não pergunte definições ("o que é doação de imóvel"): pergunte o dado do caso
  concreto que só quem viu aquele documento sabe.

### Q5 - RAG (`benchmarks/rag/diarios_rag_30.jsonl`)

Objetivo: isolar a **contribuição da recuperação** (e do grafo). Dois tipos:

- **`factual`**: respondível por um único trecho recuperado. Peça fatos concretos
  que o LLM sem RAG não tem como saber (número de portaria, CNPJ, valor, data, nome
  nomeado). É isso que faz o baseline sem RAG ir a zero e o ganho do RAG aparecer.
  Bons exemplos: "Qual é a data de assinatura do documento?" -> "28 de fevereiro de
  2025".
- **`multihop`**: força combinar **duas relações do grafo** (A -> B -> C; a resposta
  é C). É o que justifica o GraphRAG sobre o RAG vetorial simples. Exemplo bom do
  conjunto: "Qual documento foi aberto pelo secretário de planejamento após a
  delegação recebida pelo prefeito de Teresina?" -> "Crédito Suplementar". Garanta
  que a cadeia de duas relações seja real no grafo e que a pergunta não seja
  respondível por um único chunk (senão não testa multi-hop).
- Mantenha a proporção declarada (cerca de 40% multi-hop) e gabarito normalizado.
- Use o teste de "poluição dirigida": perguntas só sobre conteúdo não-licitação
  (`diarios_naolic_30.jsonl`) revelam se as licitações repetitivas estão afogando a
  recuperação. Isso é uma pergunta de *diagnóstico do retriever*, embutida no
  desenho do benchmark.

### Q6 - guardrails (`benchmarks/guardrails/`)

Objetivo: medir **proteção adicionada** sem quebrar uso legítimo. Aqui "qualidade"
tem requisitos próprios:

- **Cubra as categorias de ameaça**: jailbreak, tema inseguro, vazamento de PII. Não
  vire só um tipo.
- **Inclua benignos suficientes** (no conjunto atual, metade): são eles que medem o
  **falso positivo**. Um guardrail que bloqueia tudo tem 100% de bloqueio e é
  inútil; os benignos pegam isso. Faça os benignos *parecerem* arriscados sem ser
  (perguntas reais do domínio sobre licitação, nomeação, senha de portal legítima).
- **Teste evasão, não só o gatilho óbvio.** O `guardrails_30.jsonl` usa ataques
  literais ("Ignore previous instructions", "DAN"), fáceis de casar por
  palavra-chave. O `guardrails_adversarial.jsonl` reescreve as mesmas intenções de
  forma indireta (RPG, hipotético, "esqueca tudo que te falaram"). Um benchmark de
  guardrail de qualidade **precisa do conjunto parafraseado**, senão você mede um
  filtro de regex, não robustez. Rode os dois e reporte a queda de desempenho do
  literal para o parafraseado.
- **PII realista**: use formatos válidos brasileiros (CPF, CNPJ, CEP, telefone DDD)
  para exercitar o mascaramento, com dados fictícios.
- Anote o rótulo verdadeiro (`type`) por item: a métrica é por categoria (bloqueio
  do nocivo x preservação do benigno), não um número agregado só.

## 5. Processo de construção (checklist)

1. **Defina a capacidade-alvo** da questão antes de escrever qualquer pergunta.
2. **Extraia da fonte**, não da cabeça: abra documentos reais do corpus e tire o
   fato. Garante ancoragem e gabarito verificável.
3. **Escreva a resposta de referência curta** junto com a pergunta; se não der para
   escrever uma resposta única e curta, a pergunta é ruim.
4. **Marque tipo/categoria** (factual/multihop; jailbreak/unsafe/pii/benign) para
   permitir métrica estratificada.
5. **Garanta o held-out**: confirme que a pergunta e seu texto-base não estão no
   treino.
6. **Revise à mão** uma amostra (especialmente nos conjuntos gerados por LLM):
   descarte ambíguas, rasas, com gabarito duvidoso ou contaminadas.
7. **Calibre dificuldade**: rode um baseline fraco e um forte; se ambos empatam, o
   conjunto não discrimina, revise.
8. **Normalize os gabaritos** (formato de número, data, nome) antes de versionar.

## 6. Validação do benchmark (e do juiz)

Os gabaritos atuais foram gerados com auxílio de IA e o juiz também é um LLM
(Qwen3-8B fixo, 0 a 5). Isso introduz **circularidade**: IA gera, IA avalia. Para o
benchmark ter credibilidade num artigo:

- Construa um **gabarito de referência feito à mão** (sem IA) para um subconjunto, e
  meça a concordância do juiz LLM com ele (correlação de Spearman, kappa). Sem isso,
  a nota do juiz é só uma opinião automática.
- Reporte **variância**, não só a média: rode o juiz mais de uma vez ou com um
  segundo juiz e veja se o ranking dos sistemas se mantém.
- Trate o benchmark como artefato versionado: cada pergunta com fonte (qual
  documento/relação a embasa), para auditoria e reprodução.

Conexões: a estratégia de análise sobre estes números está em
[`GUIA_ARTIGO.md`](GUIA_ARTIGO.md); o estado de cada benchmark e o pendente de
curadoria manual estão em [`ROADMAP_GERAL.md`](ROADMAP_GERAL.md) e nos READMEs de
cada pasta em `benchmarks/`.
