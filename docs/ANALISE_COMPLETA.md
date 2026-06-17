# Análise completa do projeto - guia para quem vem da computação (sem ML/IA)

Este documento explica o projeto, os indicadores (métricas) e as decisões para um
público de **computação que NÃO trabalha com IA / aprendizado de máquina / PLN**.
A ideia é: se você sabe programar, conhece funções, otimização, complexidade,
estruturas de dados e um pouco de probabilidade, você vai entender tudo aqui sem
precisar saber nada de "machine learning" de antemão. Usamos analogias da
computação em vez de jargão.

O projeto é o ciclo de vida completo de um LLM (modelo de linguagem grande)
especializado em dados públicos do Piauí, em 6 frentes (Q1 a Q6). Os números estão
em `results/` e `results/README.md`; aqui explicamos o que eles SIGNIFICAM.

---

## Parte 1 - Os fundamentos

### 1.1 O que é um "modelo de linguagem" (LLM)

Pense num modelo de linguagem como uma **função gigante que prevê o próximo pedaço
de texto**. Você dá um trecho ("O gato subiu no") e ela devolve uma distribuição de
probabilidade sobre qual pedaço vem a seguir ("telhado" 40%, "muro" 12%, "carro"
0.1%, ...). Só isso. Gerar texto é repetir: prever o próximo pedaço, anexar, prever
de novo.

Esses "pedaços" se chamam **tokens**. Um token é tipicamente um pedaço de palavra
(subpalavra), não exatamente uma palavra: "nomeação" pode virar ["nome", "ação"].
Existe um vocabulário fixo de tokens (no Qwen3 são ~151.936 tokens distintos). O
modelo, no fim, sempre escolhe entre esses ~152 mil símbolos.

### 1.2 O que são "parâmetros" de um modelo

Um modelo é uma função matemática `f(entrada) -> saída`, mas é uma função com
**bilhões de números ajustáveis** dentro dela. Esses números são os **parâmetros**
(também chamados "pesos"). São apenas números reais (floats) que multiplicam e somam
a entrada por dentro, em muitas camadas.

Analogia da computação: imagine uma função com bilhões de constantes. Em
`y = a*x1 + b*x2 + c*x3 + ...`, os `a, b, c` são parâmetros. Um LLM é essa ideia
levada ao extremo: bilhões de coeficientes organizados em matrizes, aplicados em
sequência (camadas) com não-linearidades no meio.

Quando dizemos **0.6B, 1.7B, 4B, 8B**, é o número de parâmetros: "B" = bilhão.
Qwen3-0.6B tem ~600 milhões de parâmetros; Qwen3-8B tem ~8 bilhões. Mais parâmetros
= a função tem mais "capacidade" de representar padrões, mas custa mais memória e
compute. Um detalhe de tamanho de arquivo: cada parâmetro ocupa 2 bytes (em
precisão bfloat16), então um modelo de 8B pesa ~16 GB só de pesos.

### 1.3 Como funciona o "treinamento"

Treinar = **ajustar os bilhões de parâmetros para que a função erre menos**. É
otimização, igual à que você conhece de minimizar uma função de custo:

1. **Forward pass:** passa um exemplo pela função e obtém a previsão.
2. **Loss (custo/erro):** compara a previsão com a resposta certa e calcula um número
   que mede o erro. Para LLM, o erro padrão é quão baixa probabilidade o modelo deu
   ao token correto (entropia cruzada, ver 2.1).
3. **Backpropagation + gradiente:** calcula a derivada do erro em relação a cada
   parâmetro (o "gradiente" = a direção em que mexer cada parâmetro reduz o erro).
   É só a regra da cadeia do cálculo, aplicada automaticamente.
4. **Passo do otimizador:** mexe cada parâmetro um pouquinho na direção que reduz o
   erro. O tamanho do passo é a **learning rate** (taxa de aprendizado). Repetir
   milhões de vezes faz o erro cair. O otimizador que usamos (AdamW) é uma versão
   esperta do gradiente descendente que mantém uma "média móvel" do gradiente por
   parâmetro para convergir melhor.

Termos que aparecem nos configs:
- **batch / batch size:** quantos exemplos processamos juntos antes de dar um passo.
  Maior = gradiente mais estável, mais memória.
- **gradient accumulation:** somar gradientes de vários mini-lotes antes do passo,
  para simular um batch grande sem caber na memória de uma vez.
- **epoch (época):** uma passada completa por todo o conjunto de treino.
- **block_size / max_length:** tamanho (em tokens) de cada exemplo que entra no
  modelo (ex.: 1024 tokens por bloco).
- **seed:** semente do gerador aleatório; fixar a seed torna o experimento
  reproduzível (mesma divisão de dados, mesma inicialização).

### 1.4 Pré-treino vs pós-treino; modelo "base" vs "instruct"

- **Pré-treino:** treinar do zero (ou continuar treinando) em MUITO texto genérico
  só para prever o próximo token. O resultado é um modelo **base**: sabe a língua,
  mas não foi ensinado a "seguir instruções" nem a conversar.
- **Pós-treino (fine-tuning):** pegar um modelo já treinado e ajustá-lo num dado
  específico ou numa tarefa. Quando o pós-treino é em pares "pergunta -> resposta no
  estilo assistente", o resultado é um modelo **instruct** (segue ordens, conversa).

Decisão-chave do projeto: nas questões 1 a 3 partimos de modelos **base**, não
instruct. Por quê? Porque queremos medir o efeito do NOSSO treino, não o alinhamento
de fábrica de um instruct. Um instruct já foi muito modificado para conversar, o que
o afasta de "texto cru" como um diário oficial (ver 4.1).

---

## Parte 2 - Os indicadores (métricas) e o que significam

### 2.1 Entropia cruzada e perplexidade (Q1)

Quando o modelo lê um texto, em cada posição ele tinha uma probabilidade para o token
que de fato veio. A **entropia cruzada** é a média de `-log(probabilidade do token
certo)`. Se o modelo dava probabilidade alta ao token certo, o `-log` é pequeno (erro
baixo); se dava probabilidade baixa, é grande (erro alto). É exatamente a mesma
entropia cruzada da teoria da informação, medida em "nats" (ou bits).

A **perplexidade (PPL)** é só `exp(entropia cruzada)`. Intuição para um CS: é o
**número efetivo de opções entre as quais o modelo hesita** a cada token. PPL = 1
seria adivinhar perfeito; PPL = 10 é como hesitar entre ~10 alternativas. Outra
leitura: PPL menor = o modelo "comprime" melhor aquele texto (modelos de linguagem e
compressores são a mesma coisa, via Shannon). **Menor é melhor.** No Q1, treinar o
modelo nos diários fez a PPL cair (ex.: 11.47 -> 6.88), ou seja, o texto dos diários
ficou menos "surpreendente" para o modelo.

Por que held-out: medimos a PPL num conjunto de documentos **que o modelo não viu no
treino** (held-out, disjunto). Se medíssemos no próprio treino, o modelo poderia ter
"decorado" e o número seria enganoso (contaminação). Held-out mede generalização.

### 2.2 Acurácia de previsão de token (Q1)

Fração das posições em que o token mais provável segundo o modelo (o argmax) é
exatamente o token correto. Maior é melhor. É uma métrica mais grosseira que a
perplexidade (só olha o topo, ignora a confiança) e é otimista em texto formulaico
(diários têm muita repetição previsível), por isso a perplexidade é a leitura
principal e a acurácia é só apoio.

### 2.3 LLM-as-judge, nota 0 a 5 (Q2, Q4, Q5)

Para tarefas de gerar resposta livre (perguntar e responder), não dá para medir
"correção" com regex ou comparação exata: a resposta certa pode ser escrita de mil
jeitos. A solução usual é **usar um modelo grande como juiz**: damos ao juiz a
pergunta, a resposta de referência e a resposta avaliada, e pedimos uma nota inteira
de 0 (errada/irrelevante) a 5 (correta e completa). Maior é melhor.

Cuidado metodológico que adotamos: usamos um **juiz fixo** (sempre o Qwen3-8B) para
comparar modelos diferentes de forma justa. Se cada modelo julgasse a si mesmo, um
modelo fraco seria também um juiz fraco/enviesado, e a comparação não valeria. (No
caso do motor 30B, como ele ocupa as 2 GPUs, ele se auto-julga - registramos isso.)

### 2.4 Transfer ratio - taxa de transferência (Q4)

Indicador que criamos para a destilação. Mede **que fração da distância entre o aluno
e o professor a destilação fechou**:

```
transfer_ratio = (nota_aluno_depois - nota_aluno_antes) / (nota_professor - nota_aluno_antes)
```

0 = a destilação não transferiu nada; 1 = o aluno alcançou o professor; valores
intermediários = fração do "gap" fechado. Ex.: SmolLM2-135M foi de 0.07 para 0.34
com o professor em 0.66, ou seja, fechou (0.34-0.07)/(0.66-0.07) = 0.46 = 46% da
distância. É uma normalização: deixa comparável a transferência entre alunos de
tamanhos diferentes (uns partem mais perto do professor que outros).

### 2.5 Taxa de bloqueio e falsos positivos (Q6)

Para guardrails (camada de segurança): de N pedidos perigosos, quantos foram
bloqueados (taxa de bloqueio, maior melhor); e de M pedidos legítimos, quantos foram
bloqueados por engano (falsos positivos, menor melhor). O dilema clássico
"helpfulness vs harmlessness" (ser útil vs ser seguro) é justamente o trade-off entre
esses dois números.

---

## Parte 3 - As 6 questões: decisão, método e leitura

### Q1 - Pré-treino contínuo (continued pre-training)

**Problema:** dado o corpus de diários oficiais, continuar o pré-treino de um modelo
base e medir antes/depois.

**Decisões e por quê:**
- Modelos **base** (Qwen3-*-Base, gemma-3-1b-pt), pelo motivo da seção 1.4.
- **Escada de tamanho** (0.6B, 1.7B): rodar vários tamanhos para ver como o ganho
  escala. Comparar modelos diferentes faz parte do método. O 4B foi tentado mas o
  full fine-tuning não cabe nas 2x L4 (ver seção 4.2): fica como limite de hardware.
- **Full-parameter:** treinamos TODOS os parâmetros (é o que a questão pede). Isso
  custa muita memória; por isso truques (ver Parte 4): em 0.6B/1.7B cabe numa GPU;
  o 4B exigiria dividir o modelo entre 2 GPUs (FSDP) e otimizador 8-bit, que nesta
  pilha de software não fecha.

**Indicadores:** perplexidade (principal), entropia cruzada, acurácia de token, num
held-out disjunto (anti-contaminação) e num conjunto de Q&A.

**Resultado:** a PPL no held-out cai em todos (ex.: 0.6B 11.47 -> 6.88; gemma-pt 9.57
-> 5.49, o melhor da escada). Achados extras: (a) um base pequeno treinado supera um
instruct muito maior sem treino, no texto de domínio; (b) podar licitações do corpus
de treino NÃO ajuda, até piora (resultado negativo honesto - ver 4.3).

### Q2 - Pós-treino SFT (Supervised Fine-Tuning)

**Problema:** gerar >= 1.000 pares de pergunta-resposta a partir dos dados de docentes
e fazer fine-tuning supervisionado; avaliar antes/depois.

**Decisões:**
- Geramos os pares com um modelo professor (Qwen3-8B) ancorado nos textos reais
  (dados sintéticos, mas factuais).
- Treinamos com **loss só na resposta** (completion-only): o modelo aprende a
  PRODUZIR a resposta, sem ser penalizado por "prever" a pergunta que já foi dada.
- Avaliamos num held-out de **recall in-domain** (perguntas novas sobre os mesmos
  textos do treino), porque queremos testar se o modelo internalizou o conhecimento.

**Indicador:** juiz 0-5 e perplexidade da resposta.

**Resultado:** o SFT baixa a perplexidade da resposta em todos os modelos (aprendeu o
estilo/conteúdo). O ganho na NOTA do juiz depende de quão fraco era o base: o
gemma-pt, fraco em seguir instrução (0.67), saltou para 1.57 (+133%); os Qwen já
respondiam razoavelmente e quase não mudaram a nota. Achado bonito: partir do
**checkpoint da Q1** (que já viu o domínio) e fazer SFT supera o SFT a partir do base
puro, nos modelos Qwen - ou seja, Q1 e Q2 se somam.

### Q3 - Pós-treino LoRA / QLoRA (PEFT)

**Problema:** repetir a Q2, mas com **PEFT** (parameter-efficient fine-tuning), e
comparar com o fine-tuning pleno.

**O que é LoRA (para um CS):** em vez de mexer nos bilhões de parâmetros originais,
você os **congela** e adiciona, ao lado de certas matrizes de pesos `W`, um pequeno
ajuste de **baixo posto** `A*B` (duas matrizes finas). Só `A` e `B` treinam. Isso
reduz os parâmetros treináveis para ~1-2% do total (analogia: em vez de reescrever um
programa inteiro, você aplica um pequeno patch). **QLoRA** = a mesma ideia com o
modelo base carregado em 4 bits (quantizado) para caber em menos memória. Ao final,
"mesclamos" o patch `A*B` de volta nos pesos para virar um modelo normal.

**Indicador:** mesmo juiz/perplexidade da Q2.

**Resultado:** **LoRA iguala ou supera o fine-tuning pleno** treinando ~1.7% dos
parâmetros (vence/empata em 5 de 6 casos). Provável motivo: com poucos exemplos
(1.000), o fine-tuning pleno de um modelo pequeno tende a "overfit/forget" (decora e
esquece), e o LoRA, por ter menos liberdade, regulariza. Isso valida o uso de PEFT:
mesma qualidade, fração do custo.

### Q4 - Destilação de conhecimento (teacher -> student)

**Problema:** definir um modelo professor (teacher) e um aluno (student), gerar um
dataset sintético, destilar o conhecimento do professor para o aluno, avaliar
antes/depois com um benchmark de 100 perguntas e analisar a transferência.

**O que é destilação:** transferir o que um modelo grande "sabe" para um pequeno.
Dois jeitos:
- **Response-based:** o professor gera respostas; o aluno é treinado (SFT) para
  imitá-las. Simples e reusa nosso pipeline.
- **Logit-KD (soft labels):** em vez de só a resposta final, o aluno aprende a
  **distribuição de probabilidades** do professor em cada token, minimizando a
  divergência de Kullback-Leibler (KL) - uma medida de "distância" entre duas
  distribuições de probabilidade. Isso transfere a "incerteza calibrada" do professor
  (o chamado dark knowledge), não só a melhor resposta. Restrição: exige que professor
  e aluno usem o MESMO vocabulário de tokens (mesma família).

**Decisões:** professor Qwen3-8B; alunos numa escada de modelos pequenos de boas
fontes (até SmolLM2-135M, ~60x menor que o professor); domínio dos diários; o aluno
responde "closed-book" (sem consultar nada).

**Indicador:** juiz 0-5, perplexidade e o **transfer ratio** (seção 2.4).

**Resultado:** houve transferência, maior nos alunos mais fracos (SmolLM2-135M 0.07
-> 0.34, fechando 46% do gap; gemma fechou 84%); a perplexidade da resposta despenca
em todos. Os dois métodos (response-based e logit-KD) empataram no aluno de 0.6B.
Ressalva honesta: o professor closed-book também não é ótimo nesse benchmark difícil,
então o "teto" é modesto; um professor com RAG (Q5) elevaria o teto.

### Q5 - RAG (Retrieval-Augmented Generation)

**Problema:** criar uma aplicação de RAG e medir o quanto ela contribui.

**O que é RAG (para um CS):** em vez de confiar só na memória (parâmetros) do modelo,
você **busca** trechos relevantes num índice e os entrega ao modelo como contexto
antes de ele responder. É como dar acesso a uma base de dados na hora da pergunta. As
peças:
- **Embeddings:** transformar cada trecho de texto num vetor de números (uma
  "coordenada" num espaço onde textos parecidos ficam perto). Usamos o modelo bge-m3.
- **Índice vetorial (FAISS):** uma estrutura que, dada a pergunta (também virada
  vetor), acha rapidamente os trechos mais próximos (busca por similaridade, tipo um
  "vizinho mais próximo" em alta dimensão).
- **Grafo de conhecimento:** além dos trechos, extraímos entidades e relações (quem
  contrata quem, quem foi nomeado) num grafo, para perguntas que exigem ligar dois
  fatos (multi-hop).
- **Agente self-reflective:** um fluxo que recupera, gera e CRITICA a própria
  resposta, podendo recuperar de novo (auto-reflexão).

**Indicador:** juiz 0-5 fixo, comparando sem RAG (baseline) vs com RAG, em 3 modos
(standard, agentic, agentic+grafo) e 3 motores.

**Resultado:** o RAG ajuda muito (baseline ~1.1 -> RAG ~2.7). O maior salto vem da
recuperação simples (standard); grafo/agente quase não melhoram num motor forte
(a recuperação "satura"). Os exemplos qualitativos mostram o padrão: **sem RAG o
modelo alucina** (inventa um artigo de lei) ou se recusa; **com RAG ele ancora** na
evidência. Também descobrimos que as licitações repetitivas "poluem" o índice
(podar ajuda a recuperação) - o oposto do que acontece no pré-treino (Q1).

### Q6 - Guardrails (camada de proteção)

**Problema:** adicionar uma camada de segurança e medir o grau de proteção.

**O que é:** filtros que inspecionam a ENTRADA (bloquear pedidos perigosos ou
tentativas de "jailbreak"/injeção de prompt) e a SAÍDA (mascarar dados pessoais -
PII: CPF, CNPJ, telefone - antes de mostrar). São componíveis (vários filtros em
sequência) e implementados por regras/regex (sem precisar de outro modelo).

**Indicador:** taxa de bloqueio das perguntas adversariais, mascaramento de PII e
falsos positivos nas benignas (seção 2.5).

**Resultado:** num benchmark de 30, a camada bloqueia/mascara 100% das adversariais e
da PII, com 0 falsos positivos nas benignas (sem a camada, 0% de proteção). Ressalva
honesta: filtros por regra pegam padrões conhecidos; ataques parafraseados evadiriam,
e um filtro por modelo generalizaria melhor (entra como mais um filtro, sem mudar a
camada).

---

## Parte 4 - Decisões transversais (que valem para todo o projeto)

### 4.1 Por que partir de modelos "base" (o imposto de alinhamento)

Medimos: um modelo instruct, em texto cru de diário, tem perplexidade MUITO maior que
o base do mesmo tamanho, ANTES de qualquer treino nosso (ex.: gemma instruct 28.2 vs
gemma base 9.6). O pós-treino de chat afasta o modelo de "prever texto cru" - um
"imposto de alinhamento". Como nas Q1-Q3 queremos medir o efeito do nosso treino,
partir do base é mais limpo. Modelos instruct ficam para inferência, RAG e destilação.

### 4.2 Truques de memória (por que o 4B é difícil)

Treinar full-parameter exige guardar, por parâmetro: o peso, o gradiente e os estados
do otimizador (no AdamW, dois valores). Em fp32 isso dá ~16 bytes/parâmetro. Para 4B
são ~64 GB, e nossas 2 placas L4 somam 46 GB. Truques (que NÃO mudam que é
full-parameter, só rearranjam memória):
- **FSDP:** dividir (shard) o modelo, gradientes e otimizador entre as 2 GPUs.
- **Gradient checkpointing:** não guardar todas as ativações; recalcular na volta
  (troca compute por memória).
- **CPU offload:** mandar partes para a RAM da CPU. Causou uma falha real (a redução
  da norma do gradiente exigia um backend de CPU que não existia) - documentada.
- **8-bit Adam:** guardar os estados do otimizador em 8 bits em vez de 32 (truque de
  memória, NÃO é PEFT: todos os pesos treinam, só os estados do otimizador são
  quantizados). Era a aposta para o 4B, mas nesta pilha de software ela não fechou.

**Desfecho do 4B (resultado negativo, ver 4.3):** mesmo com esses truques, o 4B em
full fine-tuning NÃO coube nas 2x L4. Quatro otimizadores foram testados e cada um
falhou por um motivo diferente: (1) AdamW com CPU offload quebra a norma do gradiente
por falta de backend de CPU; (2) o 8-bit Adam do bitsandbytes não tem estratégia de
sharding sob FSDP; (3) o 8-bit do torchao quebra no torch.compile; (4) o mesmo em modo
eager dá conflito de tipo (bf16 vs fp32) sob DTensor. O AdamW comum (32 bits) estoura a
memória já no cálculo da perda. Conclusão: 4B full-parameter excede 2x L4 de 22 GB com
torch 2.12 / transformers 5.11. A escada 0.6B/1.7B cobre a Q1; o 4B fica documentado
como limite de hardware (QLoRA caberia, mas é PEFT = Q3, fora do escopo full-FT da Q1).

### 4.3 Resultados negativos honestos

Nem tudo "deu certo", e isso é valioso para um trabalho sério. Exemplos: (1) a hipótese
de que podar licitações melhoraria o pré-treino (Q1) foi TESTADA e refutada (piorou);
um artefato de benchmark que parecia mostrar melhora foi identificado e corrigido. (2)
o full fine-tuning do 4B não cabe nas 2x L4 (ver 4.2): quatro otimizadores testados,
todos falham por motivos distintos. Reportar esse limite de hardware, com as quatro
tracebacks documentadas, é mais honesto que omitir o 4B. Esse rigor (testar, medir, e
reportar o resultado mesmo quando contraria a hipótese ou expõe um limite) vale mais que
só mostrar números bonitos.

### 4.4 Por que a arquitetura é "OCP/SOLID"

O código é organizado para que novos modelos, métodos de treino e métricas entrem por
**extensão** (novas classes registradas + um arquivo de configuração YAML), sem
reescrever o núcleo. Para um CS: é o Open-Closed Principle. Na prática, adicionar o
trainer de destilação (Q4) ou a camada de guardrails (Q6) foi criar uma classe nova e
registrá-la; o `scripts/train.py` não mudou. Isso torna os experimentos baratos e
reprodutíveis (tudo dirigido por config + seed).

---

## Parte 5 - Os resultados em detalhe (número a número)

Aqui percorremos TODOS os resultados das várias execuções e explicamos o que cada
número diz. A dispersão (a variação entre modelos e métodos) é parte da evidência.

### Q1.a - Pré-treino contínuo: held-out, perplexidade (menor é melhor)

| Modelo | variante | PPL antes | PPL depois | TokAcc depois |
|--------|----------|-----------|------------|---------------|
| Qwen3-0.6B-Base | base | 11.47 | 6.88 | 0.603 |
| Qwen3-1.7B-Base | base | 8.59 | 5.73 | 0.627 |
| gemma-3-1b-pt | base | 9.57 | 5.49 | 0.637 |
| gemma-3-1b-it | instruct | 28.21 | 6.87 | 0.609 |

Leitura: a PPL "depois" é sempre menor que "antes", ou seja, o pré-treino nos diários
fez o modelo prever melhor esse texto. O modelo maior parte de uma base melhor e
termina mais baixo (0.6B 6.88 -> 1.7B 5.73). O gemma de 1B termina em 5.49 (o melhor),
apesar de menor que o Qwen 1.7B: arquitetura/tokenizer importam. O gemma instruct
começa MUITO pior (28.21, o "imposto de alinhamento" da seção 4.1) e, mesmo treinado,
fica acima do gemma base (6.87 vs 5.49).

### Q1.b - Base fine-tunado vs instruct sem treino (held-out, PPL)

| Tamanho | base antes | base depois (treinado) | instruct sem treino |
|---------|------------|------------------------|---------------------|
| 0.6B | 11.47 | **6.88** | 16.30 |
| 1.7B | 8.59 | **5.73** | 11.92 |
| 4B | 7.17 | nao treinado (nao cabe nas 2x L4) | 10.02 |
| 8B | - | - | 8.17 |

Leitura: o base treinado vence o instruct do mesmo tamanho com folga (0.6B 6.88 vs
16.30). E mais: o nosso 1.7B base treinado (5.73) e até o 0.6B (6.88) batem o 8B
instruct SEM treino (8.17), um modelo 5x a 13x maior. Ou seja, adaptar ao domínio
vale mais que puro tamanho. Confirma a decisão de partir de base.

### Q1.c - Ablação: podar licitações do corpus de treino (PPL, menor melhor)

| Corpus de treino | held-out original | held-out balanceado | QA |
|------------------|-------------------|---------------------|-----|
| completo | **6.88** | **6.86** | **10.13** |
| balanceado (licitação podada) | 7.16 | 7.09 | 10.29 |

Leitura: treinar no corpus balanceado (com menos licitações) deu PPL PIOR em todos os
conjuntos. Hipótese refutada: para o pré-treino, mais texto do domínio (mesmo
repetitivo) ajuda. É o oposto do RAG (Q5.c).

### Q2 - SFT (recall in-domain, juiz 0-5 / perplexidade da resposta)

| Modelo | base juiz/ppl | SFT juiz/ppl | SFT-de-Q1 juiz/ppl |
|--------|---------------|--------------|--------------------|
| Qwen3-0.6B | 1.49 / 9.29 | 1.49 / **6.44** | 1.61 / 6.76 |
| Qwen3-1.7B | 1.88 / 7.44 | 1.89 / **5.09** | **1.99** / 5.14 |
| gemma-3-1b | 0.67 / 10.95 | **1.57** / 7.38 | 1.47 / 7.55 |

Leitura: a perplexidade da resposta cai em TODOS (o SFT ensinou o estilo/conteúdo das
respostas). A nota do juiz só sobe muito onde o base era fraco: o gemma (0.67 -> 1.57,
+133%); os Qwen já respondiam (1.5-1.9) e quase não mudaram. A coluna "SFT-de-Q1"
(partindo do modelo já pré-treinado na Q1) supera o "SFT-de-base" no juiz para o Qwen
(0.6B 1.61 vs 1.49; 1.7B 1.99 vs 1.89): Q1 e Q2 se somam.

### Q3 - LoRA vs SFT pleno (recall, juiz 0-5; LoRA treina ~1.7% dos parâmetros)

| Modelo (partida) | juiz SFT pleno | juiz LoRA |
|------------------|----------------|-----------|
| 0.6B (base) | 1.49 | **1.69** |
| 1.7B (base) | 1.89 | **2.05** |
| 1.7B (Q1) | 1.99 | **2.11** |
| gemma (base) | 1.57 | **1.67** |
| gemma (Q1) | 1.47 | **1.65** |
| 0.6B (Q1) | 1.61 | 1.60 |

Leitura: o LoRA iguala ou supera o fine-tuning pleno em 5 dos 6 casos, treinando uma
fração mínima dos parâmetros. Provável regularização (o full overfita com 1.000
exemplos). Mensagem: PEFT entrega a mesma qualidade muito mais barato.

### Q4 - Destilação (benchmark de 100, recall; juiz / ppl-resposta)

| Aluno (student) | params | base juiz | distill juiz | transfer ratio | base ppl | distill ppl |
|-----------------|--------|-----------|--------------|----------------|----------|-------------|
| SmolLM2-135M | 135M | 0.07 | 0.34 | 0.46 | 22.2 | 19.9 |
| SmolLM2-360M | 360M | 0.18 | 0.34 | 0.33 | 12.6 | 11.7 |
| Qwen2.5-0.5B | 0.5B | 0.34 | 0.46 | 0.38 | 12.4 | **6.3** |
| Qwen3-0.6B | 0.6B | 0.60 | 0.51 | base~teacher | 10.6 | **6.1** |
| gemma-3-1b | 1.0B | 0.41 | 0.62 | **0.84** | 11.3 | **4.6** |

(professor Qwen3-8B: juiz 0.66, ppl 10.6.)

Leitura: houve transferência (4 dos 5 alunos sobem no juiz). O 135M, que mal respondia
(0.07), foi a 0.34 (fechou 46% da distância para o professor); o gemma fechou 84%
(0.41 -> 0.62, quase o professor 0.66). A perplexidade da resposta despenca em todos:
os alunos absorveram a distribuição do professor. O Qwen3-0.6B já estava perto do
professor (0.60 vs 0.66), sem espaço para subir no juiz (mas a ppl caiu muito).
Comparação de métodos no 0.6B: response-based 0.51/6.12 vs logit-KD 0.50/6.51 -
empate, mesmo o logit-KD usando um professor menor (1.7B).

### Q5.a - RAG: 3 modos x 3 motores (juiz 0-5, corpus cheio)

| Motor | baseline (sem RAG) | standard | agentic (sem grafo) | agentic + grafo |
|-------|--------------------|----------|---------------------|-----------------|
| Qwen3-8B | 1.10 | **2.70** | 2.60 | 2.63 |
| gemma-3-1b-it | 0.67 | 2.07 | **2.23** | 2.03 |
| gemma-3-1b-pt (base) | 0.47 | 0.73 | 0.73 | **0.87** |

Leitura: o salto grande é do baseline para o standard (a recuperação simples é o ganho
principal: 8B 1.10 -> 2.70). Grafo/agente quase não separam do standard num motor forte
(o 8B "satura"); ajudam mais num motor fraco (gemma). O motor importa: instruct > base
(8B > gemma-it > gemma-pt). O gemma-pt base chega a PIORAR com RAG (0.73 < baseline
1.10? na verdade baseline 0.47): um modelo base fraco se confunde com o contexto.

### Q5.b - RAG: corpus cheio vs balanceado (licitações podadas)

| Motor | modo | cheio | balanceado |
|-------|------|-------|------------|
| Qwen3-8B | standard | 2.70 | **3.50** |
| Qwen3-8B | agentic+grafo | 2.63 | 3.37 |
| gemma-1b-it | standard | 2.07 | 2.40 |

Leitura: aqui, AO CONTRÁRIO da Q1, podar as licitações repetitivas AJUDA (8B standard
2.70 -> 3.50): no RAG, licitações quase iguais afogam o top-k da busca e diluem o
resto. Mas isso ajuda a RECUPERAÇÃO, não o grafo (os modos seguem próximos). Em
produção não se descartam licitações; é só um diagnóstico.

### Q6 - Guardrails (benchmark de 30; quantos tratados, com vs sem a camada)

| Tipo | n | sem guardrails | com guardrails |
|------|---|----------------|----------------|
| jailbreak (bloquear) | 5 | 0 | **5** |
| inseguro (bloquear) | 5 | 0 | **5** |
| PII na saída (mascarar) | 5 | 0 | **5** |
| benigna (passar) | 15 | 15 | 15 |

Leitura: a camada leva a proteção de 0% para 100% nas adversariais e na PII, sem
bloquear nenhuma das 15 benignas (0 falsos positivos). Ressalva: são regras, pegam
padrões conhecidos; um ataque reescrito evadiria.

## Parte 6 - Glossário rápido

- **Token:** pedaço de palavra; a unidade que o modelo lê e gera.
- **Parâmetro / peso:** um dos bilhões de números ajustáveis da função-modelo.
- **Pré-treino / fine-tuning:** treinar do zero em texto genérico / ajustar um modelo
  pronto numa tarefa ou dado específico.
- **Base / instruct:** modelo só pré-treinado / modelo também pós-treinado para
  seguir instruções e conversar.
- **Loss:** o número que mede o erro; o treino minimiza a loss.
- **Gradiente / backpropagation:** a direção para ajustar cada parâmetro; calculada
  pela regra da cadeia.
- **Learning rate:** tamanho do passo de ajuste.
- **Epoch / batch:** uma passada pelos dados / quantos exemplos por passo.
- **Perplexidade (PPL):** exp(entropia cruzada); "quantas opções" o modelo hesita por
  token; menor é melhor.
- **Entropia cruzada:** média de -log(prob do token certo); o erro de previsão.
- **Acurácia de token:** fração de tokens em que o topo do modelo acerta.
- **LLM-as-judge:** usar um modelo grande para dar nota 0-5 a uma resposta.
- **SFT:** Supervised Fine-Tuning; ajustar em pares pergunta-resposta.
- **LoRA / QLoRA:** ajustar só um pequeno "patch" de baixo posto / o mesmo com o base
  em 4 bits. PEFT = parameter-efficient fine-tuning.
- **Destilação (KD):** transferir conhecimento de um modelo grande (teacher) para um
  pequeno (student). KL = divergência entre distribuições, usada no logit-KD.
- **RAG:** Retrieval-Augmented Generation; buscar trechos num índice e dar ao modelo
  como contexto.
- **Embedding:** vetor de números que representa um texto; textos parecidos ficam
  perto. FAISS = índice para achar os vizinhos mais próximos rápido.
- **FSDP / offload / gradient checkpointing / 8-bit Adam:** truques para treinar
  modelos grandes em pouca memória, sem deixar de ser full-parameter.
- **Guardrails:** camada de filtros de segurança na entrada/saída (bloquear, mascarar
  PII, etc.).
- **PII:** dados pessoais identificáveis (CPF, CNPJ, telefone, email).
- **Held-out:** dados separados, não usados no treino, para medir generalização sem
  contaminação.
- **Seed:** semente aleatória; fixar = reprodutibilidade.

---

Números e tabelas detalhadas: `results/README.md` e os CSVs em `results/`.
