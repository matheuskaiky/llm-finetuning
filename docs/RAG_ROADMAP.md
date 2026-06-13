# Roadmap e decisoes do RAG (Q5)

Decisoes de arquitetura e proximos passos do RAG, para nao se perderem entre as
iteracoes. Resultados numericos ficam em `results/`; o journal detalhado em
`NOTAS.md`.

## Achado: licitacoes "poluem" um indice unico

Os diarios tem muitos atos de licitacao com estrutura quase identica. Num indice
unico isso afoga a recuperacao: o top-k vira varias licitacoes quase iguais e
dilui o resto. Evidencia (junho/2026): ~36 por cento do corpus e licitacao-pesado;
podar/balancear elevou o `standard` do Qwen3-8B de 2.70 para 3.50 (/5) e deu um
grafo mais rico (202 -> 411 entidades). Ou seja, a hipotese se confirmou.

Mas **balancear apagando licitacoes nao e a solucao de producao**: licitacoes sao
informacao real e valiosa. O dataset balanceado fica apenas como experimento de
diagnostico (mantido para comparacao).

## Estrategias para manter as licitacoes sem poluir (a avaliar)

1. **Indices segmentados por tipo + roteador** (ideia do usuario, recomendada como
   base). Um indice/grafo so de licitacoes e outro "geral"; um roteador
   (classificador leve por `is_licitacao`/metadados, ou um LLM) decide para qual
   mandar a pergunta; perguntas cruzadas consultam os dois e fundem o contexto.
   Permite ate um grafo com schema especializado por tipo (licitacao:
   EMPRESA -> LICITACAO -> PREFEITURA -> VALOR; geral: nomeacoes, orgaos,
   pertencimento). Nao perde nenhum documento.
2. **Dedup / colapso de near-duplicatas.** Como as licitacoes sao repetitivas,
   colapsar near-duplicatas (MinHash/LSH, ja implementado no `DocenteExtractor`)
   para um representante por "template" mantem a informacao e remove a redundancia
   que polui. Pode ser feito num indice unico.
3. **Diversidade na recuperacao (MMR).** Reordenar o top-k penalizando redundancia
   (Maximal Marginal Relevance), para o top-k nao ser cinco licitacoes quase iguais.
   Ataca o sintoma diretamente, sem mexer no dataset.
4. **Metadados + retrieval hibrido.** Taggear cada chunk com `doc_type`; filtrar ou
   dar boost por tipo conforme a intencao da pergunta.

Recomendacao: combinar (1) segmentacao + roteador com (2)/(3) dedup + MMR. Encaixa
no OCP atual: cada indice e um `Retriever`; o roteador e uma estrategia; o registro
de runners (`rag/pipelines.py`) aceita um modo novo por extensao, sem mudar o
avaliador.

## Plano futuro: usar os modelos fine-tuned (Q1/Q2/Q3/Q4) como motor do RAG

Quando Q1 (pre-treino continuo), Q2 (SFT), Q3 (LoRA/QLoRA) e Q4 (destilacao)
estiverem concluidas, testar os **checkpoints treinados** como motor do RAG (Q5) e
comparar com os base/instruct atuais:
- Q1: base com pre-treino continuo de dominio;
- Q2/Q3: modelos pos-treinados (SFT pleno e PEFT) no docentesDC;
- Q4: o **student destilado** (interessante ver se um modelo pequeno destilado
  sustenta bem o RAG, juntando custo baixo com conhecimento transferido).
Mostraria o efeito de cada fase de treino sobre o RAG (geracao mais aderente ao
dominio). E so trocar `llm.model_name` no config (o motor ja e plugavel). Fazer
apenas depois de cada questao concluida.

## Nao apagar (comparacao)

Manter os datasets, indices, configs e resultados atuais (corpus cheio e
balanceado) para comparacao. Cada nova estrategia entra como variante nova, sem
remover as anteriores.
