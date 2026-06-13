# Estrategias para manter licitacoes - avaliacao (Q5)

Teste controlado (mesmo benchmark `diarios_rag_30.jsonl`, mesmo motor Qwen3-8B,
auto-juiz consistente, modo `standard` = qualidade de recuperacao). Acerto 0-5:

| Estrategia (corpus cheio, licitacoes mantidas) | standard |
|------------------------------------------------|----------|
| full (sem estrategia)                          | 2.70 |
| full + MMR                                      | 2.47 |
| full + dedup lexical (MinHash 0.85)             | 2.70 (0 chunks removidos) |
| full + dedup + MMR                              | 2.47 |
| indice balanceado (licitacoes podadas), MESMO benchmark | 2.23 |

Conclusoes:
- Em comparacao justa (mesmo benchmark), NENHUMA estrategia supera o full simples
  (2.70). MMR piora (penaliza relevancia em favor de diversidade que estas
  perguntas factuais nao querem); o dedup lexical a 0.85 nao remove nada (o
  boilerplate de licitacao vem intercalado com especificos variaveis, entao
  shingles de 5 palavras nao casam).
- O ganho aparente do balanceamento (3.50) era ARTEFATO do benchmark: aquele run
  usou um benchmark proprio (mais facil). No mesmo benchmark, o indice balanceado
  fica PIOR (2.23), pois perde documentos necessarios.
- Ou seja, a "poluicao" por licitacoes NAO prejudica a recuperacao de forma
  mensuravel neste teste. Melhor estrategia para manter licitacoes: mante-las todas
  num indice unico, recuperacao simples (sem MMR, sem dedup).

Ressalva: o benchmark mistura factual+multi-hop gerado do indice cheio; ele pode
nao estressar o caso exato de "pergunta nao-licitacao afogada por licitacoes". Um
benchmark dirigido (so perguntas nao-licitacao) seria necessario para detectar esse
caso especifico, se existir.

MMR e dedup ficam implementados e testados (toggles por config: `agent.use_mmr`,
`chunking.dedup_near`), desligados por padrao por nao ajudarem aqui.
