# Benchmark de pós-treino (Q2/Q3 - docentes)

Avaliação da fase de **pós-treino** (Questão 2, SFT, e Questão 3, LoRA/QLoRA) sobre
o dataset oficial `vickminari/docentesDC`. Mede a qualidade do modelo **antes e
depois** do fine-tuning supervisionado. Não faz parte da avaliação da Q1 (que é só
o antes/depois do pré-treino, em `benchmarks/pre_treino/`).

## Conjuntos (gerados, não versionados)

Os pares são derivados do `docentesDC` por `scripts/build_sft_pairs.py`, ancorando
cada `{instruction, input?, output}` em um trecho do texto-fonte:

- **Treino** - 1.000+ pares para o SFT (Q2) e o LoRA/QLoRA (Q3, mesmo conjunto).
- **Held-out** - 150 pares disjuntos do treino, para medir generalização.
- **Recall** - 150 perguntas cujas respostas estão nos textos-fonte, excluindo as
  perguntas de treino, para medir o conhecimento efetivamente transferido.

Os arquivos ficam em `data/processed/sft/` (git-ignored, regeneráveis pelo script),
ex.: `docentes_sft_heldout.jsonl`.

## Antes e depois (do pós-treino)

O `eval_sft.py` recebe vários modelos numa só corrida (base e checkpoint treinado) e
gera uma tabela comparando-os no mesmo held-out, isolando o efeito do SFT/LoRA. As
pastas `results/antes/` e `results/depois/` ficam reservadas para saídas JSON
separadas por momento, quando geradas.

## Métricas

Avaliado por `scripts/eval_sft.py`: as métricas intrínsecas sobre a resposta
(perplexidade da resposta sob teacher-forcing, entropia cruzada, acurácia de token)
mais um LLM-as-judge (0 a 5, Qwen3-8B fixo em `--judge-model`, decodificação gulosa).
A Q3 também compara custo (VRAM e tempo) e fração de parâmetros treinados contra o
SFT pleno da Q2.

## Como rodar

```bash
# 1. Gerar os pares (treino + held-out + recall) a partir do docentesDC
python scripts/build_sft_pairs.py

# 2. Treinar (SFT na Q2; mesma config com bloco peft para LoRA/QLoRA na Q3)
python scripts/train.py --config configs/sft_docentes_qwen3_0p6b.yaml

# 3. Comparar base vs treinado no mesmo held-out
python scripts/eval_sft.py --models models/Qwen3-0.6B outputs/sft_qwen3_0p6b \
  --heldout data/processed/sft/docentes_sft_heldout.jsonl --out results/q2_sft.csv
```

Os números consolidados (Q2: `q2_sft.csv`; Q3: `q3_lora.csv`) e a síntese estão em
[`results/README.md`](../../results/README.md).
