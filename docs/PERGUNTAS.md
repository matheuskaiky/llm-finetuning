Quantidades **de fato usadas** em cada parâmetro (este ponto costuma ser mal lido, então
está explícito):

| Etapa | Dado | Quantidade | Arquivo |
|-------|------|-----------|---------|
| Q1 treino (pré-treino contínuo) | texto de diário | **2.000 documentos** (~7.802 blocos de 1024 tokens, da ordem de **8M tokens** treinados) | `data/processed/diarios_txt/` |
| Q1 held-out (avaliação) | texto de diário inédito | **150 documentos** disjuntos do treino | `data/processed/diarios_heldout.jsonl` |
| Q1 benchmark P&R | perguntas/respostas conceituais sobre o domínio (geradas com auxílio de IA) | **33 perguntas** | `benchmarks/pre_treino/diarios_qa.jsonl` |
| Q2 SFT | pares de instrução | **1.000 pares** | `data/processed/sft/docentes_sft_train.jsonl` |
| Q2/Q3 recall (avaliação) | pares in-domain | **150 pares** (sem as perguntas de treino) | (gerado por `build_sft_pairs.py --recall`) |
| Q3 LoRA | mesmos pares de SFT | **1.000 pares** | idem Q2 |
| Q4 destilação (principal) | Q&A sintético do professor | **1.200 pares** | `data/processed/distill/diarios_distill_train.jsonl` |
| Q4 comparação de professores | Q&A sintético | **400 pares por professor** (orçamento fixo) | `data/processed/distill/diarios_distill_t*_train.jsonl` |
| Q4 benchmark | recall in-domain | **100 perguntas** | `data/processed/distill/diarios_distill_recall.jsonl` |
| Q5 RAG | índice + benchmark | índice sobre subconjunto do córpus; **30 perguntas** (factual + multi-hop) | `benchmarks/rag/diarios_rag_30.jsonl` |
| Q6 guardrails | benchmark | **30 perguntas** (10 adversariais, 5 PII, 15 benignas) + **15 parafraseadas** | `benchmarks/guardrails/guardrails_30.jsonl`, `guardrails_adversarial.jsonl` |