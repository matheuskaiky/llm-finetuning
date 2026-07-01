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
- **Formato compatível com a capacidade do modelo.** O formato da avaliação tem de
  casar com o que o modelo sabe fazer. Modelos base/foundation são treinados só para
  prever o próximo token, não para seguir instruções nem responder perguntas; cobrar
  P&R aberta deles é errático, porque mistura "sabe o fato" com "entende o formato de
  pergunta", e penaliza o base por algo que ele nunca aprendeu. Para esses modelos,
  prefira formatos nativos de modelagem de linguagem (Cloze/completar lacunas,
  perplexidade, likelihood de continuações). Reserve a P&R aberta para os modelos
  instruct, ou use os dois formatos lado a lado (ver Q1).

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

### Q1 - dois formatos: P&R e Cloze sobre os diários

Objetivo: medir o efeito do **pré-treino contínuo** no domínio (antes vs depois).

A Q1 usa **dois benchmarks complementares**, aplicados aos modelos **base e
instruct**:

- **P&R** (`benchmarks/pre_treino/diarios_qa.jsonl` e o gabarito a mão
  `benchmarks/a-mao/Q1 benchmark P&R.csv`): pergunta aberta com resposta curta.
- **Cloze / completar lacunas** (novo): uma frase ancorada num ato real do diário com
  um trecho saliente mascarado; o modelo deve recuperar o trecho.

Por que dois. A P&R aberta é errática em modelos **base**: eles só preveem o próximo
token, não foram treinados para seguir instrução nem responder pergunta, então a nota
mistura "sabe o fato" com "entende o formato". O Cloze é um formato nativo de
modelagem de linguagem (prever o token mascarado), justo tanto para o base quanto para
o instruct. Rodar os dois formatos nos dois tipos de modelo permite separar o efeito
do domínio (que o Cloze isola) do efeito de formato/instrução (visível na diferença
P&R base vs instruct).

Regras comuns aos dois formatos:

- A pergunta/lacuna deve depender da *forma e do conteúdo* dos atos do corpus, não da
  definição genérica do instituto jurídico. Esse é o ponto que separa Q1 de um
  quiz de direito administrativo.
- Resposta curta e factual. A métrica principal é intrínseca (perplexidade, entropia,
  acurácia de token sobre texto de domínio); os conjuntos de P&R e Cloze são sondas
  complementares.
- Cubra as classes de ato reais: decreto, portaria, edital de licitação, nomeação,
  orçamento/crédito suplementar, regime de pessoal. Evite concentrar em uma só.
- Held-out por construção: as perguntas/lacunas são escritas à mão, fora do corpus de
  treino. Acompanhe sempre o held-out de **texto de diário inédito** (150 docs
  disjuntos): é ele que mede generalização sem depender do juiz.

#### Específico do Cloze (Opção A)

O segredo de um bom Cloze para pré-treino contínuo é eliminar o tom de
"interrogatório" (pergunta e resposta direta) e focar na **continuidade natural do
texto**. No pré-treino avaliamos a capacidade do modelo de prever os próximos tokens
pela probabilidade de escrita, então a instância deve parecer uma frase do próprio
diário cortada no meio, não uma pergunta.

Princípios de redação da lacuna:

- Mascare um trecho **saliente e unicamente recuperável** (um nome, valor, número de
  ato, data, cargo), não palavra funcional ("de", "para") nem algo adivinhável pelo
  contexto sem conhecer o documento.
- O `context` é o texto truncado **exatamente antes** do dado a prever, terminando de
  forma que a continuação natural seja o `target`. Dê contexto suficiente em volta
  para desambiguar, mas não a ponto de entregar a resposta.
- A lacuna deve ter **uma única** resposta certa dado o contexto. Mantenha o `target`
  curto para o acerto ser nítido.
- Use os mesmos documentos-fonte da P&R quando possível, para os dois formatos
  medirem o mesmo conhecimento por ângulos diferentes.

Estrutura de colunas do dataset (para rodar automatizado, com script próprio ou
integrando ao LM Eval Harness):

| Coluna | Descrição | Exemplo |
| --- | --- | --- |
| `id` | Identificador único da instância. | `1` |
| `context` | Texto truncado exatamente antes do dado a prever (a deixa para o modelo). | `"No contrato de prestação de serviços temporários do Município de Simões de maio de 2025, a servidora Maria Alice de Carvalho Serio foi contratada para exercer a função exata de"` |
| `target` | Final exato da frase (gabarito de predição). | `"Auxiliar de Atividade Educacional 1"` |
| `municipio` | Metadado para filtrar/analisar desempenho por região. | `"Simões"` |
| `tipo_documento` | Metadado para ver onde o modelo erra mais (contrato, portaria, diário, licitatório). | `"Contrato Temporário"` |
| `arquivo_origem` | Nome do `.txt` de onde o dado saiu (auditoria de contaminação e checagem de fato). | `9a2a8026e2b9531d4f2e24d182e8971c.txt` |

Que dados priorizar no `target`. O objetivo é testar a inteligência contextual e de
domínio aplicada à burocracia municipal, ou seja, exigir que o modelo tenha de fato
internalizado os documentos do Piauí no treino, não chutado termos comuns. Priorize:

- **Entidades nominais raras (nomes próprios e cargos específicos).** Modelos
  genéricos sabem o que é um "Prefeito", mas só o modelo treinado com os diários
  locais sabe que "Thais Meneses Freitas foi nomeada para exercer o cargo de [Chefe de
  Departamento de Assistência Farmacêutica]".
- **Valores financeiros exatos (dotações, contratos e dispensas).** Valores monetários
  mudam de um município para outro; acertar que a dispensa de Paulistana para
  dedetização foi de exatamente "R$ 61.600,00" é um bom teste de retenção factual.
- **Siglas, setores e leis combinados.** Força o modelo a ligar o contexto jurídico ao
  valor, ex.: "em conformidade com os 15% da LC 141/2012, o valor listado como Despesa
  Mínima a ser Aplicada em ASPS é de [R$ 108.675,60]".

Como medir o sucesso. Duas abordagens de métrica, complementares:

1. **Exact Match (correspondência exata).** O texto gerado pelo modelo após o
   `context` tem de ser idêntico ao `target`. Boa para valores, datas e CNPJs (com a
   normalização de OCR descrita abaixo).
2. **Perplexidade (PPL) / log-likelihood.** Em vez de deixar o modelo gerar livre,
   passa-se a frase completa (`context` + `target`) e mede-se quão "surpreso" o modelo
   fica. Quanto menor a perplexidade (ou maior o log-likelihood do `target`), melhor o
   modelo aprendeu aquele fato no pré-treino contínuo. É a métrica que funciona igual
   para base e instruct, por não depender de geração nem de seguir instrução.

Cuidados de fonte (valem para P&R e Cloze):

- **Um documento de diário cobre vários municípios.** No córpus, cada `.txt` agrega
  atos de prefeituras diferentes na mesma página/edição. Por isso, no gabarito feito
  a mão, é normal e correto haver mais de uma pergunta/lacuna apontando o mesmo
  arquivo-fonte; não é duplicata. Registre o arquivo de origem (coluna `arquivo`) para
  auditoria, e trate a anticontaminação por documento.
- **Conte com o ruído de OCR ao avaliar.** O texto-fonte tem erros de OCR (ex.:
  `30.000` vira `30,00o`), então pontuar por correspondência exata de string
  subconta acertos. Normalize a resposta (remova `R$`, separadores de milhar/decimal,
  tolere `o` por `0`) ou use casamento tolerante. No Cloze, prefira mascarar trechos
  menos sujeitos a ruído de OCR, ou normalize o alvo do mesmo jeito.

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
