"""Generate notebooks/graficos_resultados.ipynb: consolidated analysis of all Q1-Q6 results.

Run from the repo root with the project venv. This script only assembles the notebook
cells; execution/output embedding is done separately with nbconvert/nbclient.

Layout: one section per question, each opening with a short report-style note on the
concepts (held-out, PPL, cross-entropy, token accuracy, P&R/QA, judge, OOD, transfer
ratio, hit-rate@k, RAG modes), followed by the charts. A cross-cutting synthesis closes.
"""

from __future__ import annotations

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------- title / glossary
md(
    """
# Resultados consolidados (Q1-Q6)

Leitura unificada de todos os resultados em `results/`, uma seção por questão e uma
síntese transversal ao final. Cada seção abre com uma nota curta sobre os conceitos
usados e o que cada gráfico mostra. O notebook é gerado por
`scripts/_build_results_notebook.py` e lê os CSVs do diretório `results/`.

## Como ler os gráficos

- **Juiz (LLM-as-judge).** Um LLM fixo (Qwen3-8B) atribui nota de 0 a 5 a cada resposta
  por corretude factual e aderência à instrução. O juiz é o mesmo para todos os modelos,
  então as notas são comparáveis entre si (maior é melhor).
- **Perplexidade (PPL).** Mede o quanto o modelo hesita ao prever o próximo token; é o
  exponencial da entropia cruzada. Menor é melhor. Só compara dentro da mesma família de
  tokenizador: o GPT-2, com vocabulário em inglês, não se compara ao Qwen ou ao Gemma.
- **Entropia cruzada (CE).** A mesma grandeza da PPL, em nats: a média de -log da
  probabilidade que o modelo dá ao token correto. Menor é melhor.
- **Acurácia de previsão de token.** Fração de posições em que o token mais provável do
  modelo coincide com o token correto. Maior é melhor; em texto formulaico (atos, editais)
  tende a ser otimista, por isso a PPL costuma ser mais informativa.
- **Held-out.** Conjunto separado do treino: o modelo nunca viu esses textos. Avaliar nele
  mede generalização, não memorização.
- **Limite de hardware.** 2x L4 (24 GB cada). O 4B/8B em fine-tuning completo e o motor de
  RAG de 31B limpo ficam de fora por memória, não por qualidade.
"""
)

# ---------------------------------------------------------------- header / helpers
code(
    '''
import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    HAS_SNS = True
except Exception:
    HAS_SNS = False

try:
    from adjustText import adjust_text
    HAS_ADJUST = True
except Exception:
    HAS_ADJUST = False

RESULTS = "results" if os.path.isdir("results") else "../results"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "figure.autolayout": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.grid": True,
    "grid.alpha": 0.3, "font.size": 9.5,
})

# qualitative palette by role (high contrast between depois/noft is intentional)
C = {"base": "#9e9e9e", "antes": "#9e9e9e", "depois": "#1f77b4", "noft": "#ff7f0e",
     "instruct": "#d62728", "sft": "#2ca02c", "lora": "#ff7f0e", "distill": "#9467bd",
     "rag": "#17becf", "good": "#2ca02c", "bad": "#d62728", "neutral": "#7f7f7f"}
# legendas em português para as condições do Q1
COND_PT = {"antes": "base (antes)", "depois": "base treinado (depois)", "noft": "instruct (sem treino)"}
# rotulos das tres metricas intrinsecas da Q1
Q1_METRICS = [("ppl", "perplexidade (menor melhor)"),
              ("ce", "entropia cruzada (menor melhor)"),
              ("tokacc", "acurácia de token (maior melhor)")]

def cols(keys, default="#1f77b4"):
    """Color list for a set of column keys, never None (pandas rejects None)."""
    return [C.get(k, default) for k in keys]

def text_on(rgba):
    """Black or white text for readability over a given cell color (luminance)."""
    r, g, b = mcolors.to_rgb(rgba)[:3]
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else "white"

def annot_heat(ax, M, im, fmt="{:.2f}"):
    """Annotate a heatmap with per-cell contrasting bold text."""
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=8, fontweight="bold", color=text_on(im.cmap(im.norm(v))))

def load(name):
    p = os.path.join(RESULTS, name)
    return pd.read_csv(p) if os.path.exists(p) else None

def skip(name):
    print(f"[pulado] {name} ainda não existe")

def params_to_m(s):
    """'0.6B'/'124M'/'1.0B' -> millions (float)."""
    if not isinstance(s, str):
        return np.nan
    m = re.match(r"([\\d.]+)\\s*([BM])", s.strip(), re.I)
    if not m:
        return np.nan
    v = float(m.group(1))
    return v * 1000 if m.group(2).upper() == "B" else v

def bar_labels(ax, fmt="{:.2f}", fontsize=8):
    for c in ax.containers:
        ax.bar_label(c, fmt=fmt, fontsize=fontsize, padding=2)

def load_rag_scores(fname, modes=("baseline", "standard", "agentic_graph")):
    """Per-question RAG score columns from a compare/student/engine CSV."""
    df = load(fname)
    if df is None:
        return None
    cs = {m: f"score_{m}" for m in modes if f"score_{m}" in df.columns}
    out = df[[c for c in ["idx", "type"] if c in df.columns] + list(cs.values())].copy()
    out = out.rename(columns={v: k for k, v in cs.items()})
    return out

print("results dir:", RESULTS, "| CSVs:", len(glob.glob(os.path.join(RESULTS, "*.csv"))))
'''
)

# ================================================================ Q1
md(
    """
## 1. Q1 - Pré-treino contínuo

O pré-treino contínuo segue treinando um modelo **base** (só pré-treino, sem alinhamento
de chat) no córpus dos diários, sem mudar a tarefa: o objetivo é apenas fazer o modelo
prever melhor o próximo token de texto oficial. A avaliação é antes/depois do treino, com
as três métricas intrínsecas (perplexidade, entropia cruzada e acurácia de previsão de
token), em dois conjuntos:

- **Held-out de diário:** texto de diário inédito, da mesma distribuição do treino. Mede a
  adaptação ao domínio.
- **Benchmark de P&R (perguntas e respostas):** 33 pares conceituais escritos à mão sobre
  o domínio (atos, licitações, orçamento, pessoal), mais distantes do texto cru. É o
  benchmark de no mínimo 25 perguntas pedido na Q1. Os modelos **instruct sem treino**
  entram aqui como referência, para mostrar o imposto que o alinhamento de chat cobra em
  texto governamental cru.

A escolha de partir de modelos base (e não instruct) é deliberada: o "antes" não pode já
saber seguir instruções por motivos alheios ao nosso treino. O GPT-2 (vocabulário em
inglês) entra como controle de arquitetura, com escala de perplexidade própria. Por fim,
o **OOD** (out-of-distribution) usa texto dos docentes para medir esquecimento: se a
perplexidade fora do domínio piora muito após o pré-treino, houve esquecimento
catastrófico.
"""
)
md("### 1.1 Held-out de diário - as tres metricas, antes/depois e instruct sem treino (Qwen/Gemma)")
code(
    '''
# held-out: ppl, ce e acuracia antes/depois (base) e instruct sem treino (noft), por familia
df = load("q1_base_vs_instruct.csv")
if df is None: skip("q1_base_vs_instruct.csv")
else:
    d = df[(df.eval_set == "heldout") & (df.family.isin(["qwen3", "gemma3"]))].copy()
    d["lbl"] = d["family"] + " " + d["params"]
    order = ["antes", "depois", "noft"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k, (col, ylab) in enumerate(Q1_METRICS):
        piv = d.pivot_table(index="lbl", columns="condition", values=col, aggfunc="first")
        cur = [c for c in order if c in piv.columns]
        piv[cur].plot(kind="bar", ax=ax[k], color=cols(cur), legend=False)
        ax[k].set_ylabel(ylab); ax[k].set_xlabel("")
        ax[k].set_title("Q1 held-out - " + col.upper())
        ax[k].tick_params(axis="x", rotation=0)
    ax[0].legend([COND_PT.get(c, c) for c in order], title="condição", fontsize=8)
    plt.tight_layout(); plt.show()
'''
)
md("### 1.2 Benchmark de P&R - as tres metricas, antes/depois e instruct sem treino (Qwen/Gemma)")
code(
    '''
# benchmark de P&R (QA conceitual): mesmas tres metricas, base antes/depois vs instruct
df = load("q1_base_vs_instruct.csv")
if df is None: skip("q1_base_vs_instruct.csv")
else:
    d = df[df.eval_set == "qa"].copy()
    order = ["antes", "depois", "noft"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k, (col, ylab) in enumerate(Q1_METRICS):
        piv = d.pivot_table(index="params", columns="condition", values=col, aggfunc="first")
        piv = piv.reindex([x for x in ["0.6B", "1.0B", "1.7B"] if x in piv.index])
        cur = [c for c in order if c in piv.columns]
        piv[cur].plot(kind="bar", ax=ax[k], color=cols(cur), legend=False)
        ax[k].set_ylabel(ylab); ax[k].set_xlabel("tamanho")
        ax[k].set_title("Q1 P&R - " + col.upper())
        ax[k].tick_params(axis="x", rotation=0)
    ax[0].legend([COND_PT.get(c, c) for c in order], title="condição", fontsize=8)
    plt.tight_layout(); plt.show()
'''
)
md("### 1.3 GPT-2 (controle de arquitetura) - as tres metricas no held-out e no P&R")
code(
    '''
# GPT-2 tem escala de PPL propria (vocabulario ingles): held-out e P&R, antes/depois
g = load("q1_gpt2.csv")
if g is None: skip("q1_gpt2.csv")
else:
    sets = [("heldout", "held-out"), ("qa", "P&R")]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
    for ri, (es, rlab) in enumerate(sets):
        d = g[g.eval_set == es].copy()
        d["lbl"] = d["model"] + " (" + d["params"] + ")"
        for ci, (col, ylab) in enumerate(Q1_METRICS):
            sub = d.set_index("lbl")[[col + "_antes", col + "_depois"]]
            sub.columns = ["antes", "depois"]
            sub.plot(kind="bar", ax=ax[ri, ci], color=[C["antes"], C["depois"]],
                     legend=(ri == 0 and ci == 0))
            ax[ri, ci].set_title(f"GPT-2 {rlab} - {col.upper()}")
            ax[ri, ci].set_xlabel(""); ax[ri, ci].set_ylabel(ylab if ci == 0 else "")
            ax[ri, ci].tick_params(axis="x", rotation=0)
    plt.tight_layout(); plt.show()
'''
)
md("### 1.4 Escala vs qualidade - perplexidade e acuracia depois do treino (todas as familias)")
code(
    '''
# escala: PPL depois e acuracia depois vs tamanho (held-out)
df = load("q1_base_vs_instruct.csv"); g = load("q1_gpt2.csv")
rows = []
if df is not None:
    d = df[(df.eval_set == "heldout") & (df.condition == "depois")]
    for _, r in d.iterrows():
        rows.append((r["family"], r["params"], r["ppl"], r["tokacc"]))
if g is not None:
    d = g[g.eval_set == "heldout"]
    for _, r in d.iterrows():
        rows.append(("gpt2", r["params"], r["ppl_depois"], r["tokacc_depois"]))
if not rows:
    skip("q1_base_vs_instruct.csv / q1_gpt2.csv")
else:
    d = pd.DataFrame(rows, columns=["family", "params", "ppl", "tokacc"])
    d["m"] = d["params"].map(params_to_m); d = d.dropna(subset=["m"]).sort_values("m")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    markers = {"qwen3": "o", "gemma3": "s", "gpt2": "^"}
    for fam, grp in d.groupby("family"):
        mk = markers.get(fam, "D")
        ax[0].scatter(grp["m"], grp["ppl"], s=90, marker=mk, label=fam, zorder=3)
        ax[1].scatter(grp["m"], grp["tokacc"], s=90, marker=mk, label=fam, zorder=3)
        for _, r in grp.iterrows():
            ax[0].annotate(r["params"], (r["m"], r["ppl"]), fontsize=7, xytext=(5, 4), textcoords="offset points")
            ax[1].annotate(r["params"], (r["m"], r["tokacc"]), fontsize=7, xytext=(5, 4), textcoords="offset points")
    ax[0].set_ylabel("PPL depois (menor melhor)"); ax[0].set_title("Q1 - escala vs perplexidade")
    ax[1].set_ylabel("acurácia de token depois (maior melhor)"); ax[1].set_title("Q1 - escala vs acurácia")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("parâmetros (M, escala log)"); a.legend(title="família", fontsize=8)
    ax[0].annotate("GPT-2 em escala própria (vocabulário inglês)", (0.02, 0.92),
                   xycoords="axes fraction", fontsize=7, color="gray")
    plt.tight_layout(); plt.show()
'''
)
md("### 1.5 Esquecimento (OOD) e ablacao de licitacao")
code(
    '''
# esquecimento: delta PPL OOD (depois - antes); >0 piora. Ablacao: corpus cheio vs podado
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
fg = load("q1_forgetting.csv")
if fg is not None:
    ood = fg[fg.eval_set == "ood_docentes"].copy().sort_values("delta_ppl")
    colors = [C["bad"] if v > 0 else C["good"] for v in ood["delta_ppl"]]
    ax[0].bar(ood["model"] + " (" + ood["params"] + ")", ood["delta_ppl"], color=colors)
    ax[0].axhline(0, color="black", lw=0.8)
    ax[0].set_ylabel("delta PPL OOD (depois - antes)")
    ax[0].set_title("Q1 - esquecimento em texto fora do domínio (>0 = piora)")
    plt.setp(ax[0].get_xticklabels(), rotation=90, ha="center")
else:
    ax[0].set_title("q1_forgetting.csv ausente")
lic = load("q1_balanceamento_licitacao.csv")
if lic is not None:
    d = lic[(lic.eval_set == "heldout_orig") & (lic.condition == "depois")].copy()
    piv = d.set_index("train_corpus")["ppl"].sort_values()
    ax[1].bar(piv.index, piv.values, color=C["depois"])
    ax[1].set_ylabel("PPL held-out"); ax[1].set_title("Q1 - corpus cheio vs podado (licitação)")
    ax[1].tick_params(axis="x", rotation=0)
    for i, v in enumerate(piv.values): ax[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
else:
    ax[1].set_title("q1_balanceamento_licitacao.csv ausente")
plt.tight_layout(); plt.show()
'''
)

# ================================================================ Q2/Q3
md(
    """
## 2. Q2 (SFT) e Q3 (LoRA) - pós-treino de instrução

O pós-treino ensina o modelo a seguir instruções, usando pares pergunta/resposta gerados
do `docentesDC`. Duas técnicas:

- **SFT pleno:** ajuste fino supervisionado que atualiza todos os pesos.
- **LoRA (PEFT):** treina apenas matrizes de baixo posto (cerca de 1,7% dos parâmetros) e
  congela o resto. Mais barato em memória e tempo.

A avaliação usa um **held-out de recall in-domain**: perguntas novas sobre os mesmos
textos-fonte do treino, sem repetir as perguntas de treino. Duas medidas:

- **Juiz (0-5):** corretude e aderência à instrução, avaliadas pelo LLM fixo.
- **Perplexidade da resposta:** perplexidade teacher-forced sobre a resposta de
  referência. Separa o aprendizado mesmo quando a geração gulosa de um modelo pequeno
  ainda erra um fato pontual (o juiz fica baixo, mas a perplexidade cai).

O experimento A/B compara partir do modelo base ou do checkpoint do pré-treino (Q1), para
ver se Q1 e o pós-treino se somam.
"""
)
md("### 2.1 Panorama - leaderboard do pos-treino (SFT pleno vs LoRA, recall n=150)")
code(
    '''
# leaderboard justo: cada barra e um (modelo, metodo, ponto de partida) no mesmo recall e juiz
sft = load("q2_sft.csv"); lora = load("q3_lora.csv")
rows = []
if sft is not None:
    s = sft[(sft.eval_set == "recall") & (sft.condition == "sft")].copy()
    s["label"] = s["model"] + " - SFT(" + s["start"] + ")"; s["method"] = "SFT pleno"
    rows.append(s[["label", "params", "mean_judge", "method"]])
if lora is not None:
    l = lora[lora.eval_set == "recall"].copy()
    l["label"] = l["model"] + " - LoRA(" + l["start"] + ")"; l["method"] = "LoRA"
    rows.append(l[["label", "params", "mean_judge", "method"]])
if not rows:
    skip("q2_sft.csv / q3_lora.csv")
else:
    lb = pd.concat(rows, ignore_index=True).dropna(subset=["mean_judge"])
    lb = lb.groupby(["label", "method"], as_index=False)["mean_judge"].mean()  # uma barra por (modelo, metodo)
    lb = lb.sort_values("mean_judge")
    colors = [C["sft"] if m == "SFT pleno" else C["lora"] for m in lb["method"]]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.34 * len(lb))))
    ax.barh(lb["label"], lb["mean_judge"], color=colors)
    ax.set_xlabel("juiz 0-5 (maior melhor)")
    ax.set_title("Pós-treino docentes (recall) - SFT pleno vs LoRA")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C["sft"], label="SFT pleno"), Patch(color=C["lora"], label="LoRA")],
              loc="lower right")
    for y, v in enumerate(lb["mean_judge"]): ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=8)
    plt.tight_layout(); plt.show()
'''
)
md("### 2.2 Q2 SFT - juiz: base vs SFT vs SFT iniciado no checkpoint da Q1")
code(
    '''
df = load("q2_sft.csv")
if df is None: skip("q2_sft.csv")
else:
    d = df[df.eval_set == "recall"].copy()
    base = d[d.condition == "antes"].groupby("model")["mean_judge"].first()
    sft0 = d[(d.condition == "sft") & (d.start == "base")].groupby("model")["mean_judge"].first()
    sftq1 = d[(d.condition == "sft") & (d.start == "q1")].groupby("model")["mean_judge"].first()
    piv = pd.DataFrame({"base": base, "SFT": sft0, "SFT de Q1": sftq1}).dropna(how="all")
    piv = piv.loc[piv.index.intersection(["Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "gemma-3-1b-pt",
                                          "gpt2", "gpt2-medium", "gpt2-large"])]
    ax = piv.plot(kind="bar", color=[C["base"], C["sft"], C["depois"]], figsize=(11, 4))
    ax.set_ylabel("juiz 0-5"); ax.set_title("Q2 - SFT: base vs SFT vs SFT iniciado em Q1")
    ax.tick_params(axis="x", rotation=90); bar_labels(ax); plt.tight_layout(); plt.show()
'''
)
md("### 2.3 Q3 LoRA vs SFT pleno - juiz e vantagem do LoRA")
code(
    '''
lora = load("q3_lora.csv"); sft = load("q2_sft.csv")
if lora is None or sft is None: skip("q3_lora.csv / q2_sft.csv")
else:
    l = lora[lora.eval_set == "recall"].copy()
    l["key"] = l["model"] + "/" + l["start"]
    lj = l.groupby("key")["mean_judge"].first()
    sm = sft[(sft.eval_set == "recall") & (sft.condition == "sft")].copy()
    pairs = l[["model", "start", "key"]].drop_duplicates()
    pairs["lora"] = pairs["key"].map(lj)
    def sft_match(model, start):
        cand = sm[(sm["params"] == l[l.model == model]["params"].iloc[0]) & (sm["start"] == start)]
        return cand["mean_judge"].iloc[0] if len(cand) else np.nan
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    plot = pairs.dropna(subset=["lora"]).copy()
    plot["sft"] = [sft_match(m, st) for m, st in zip(plot["model"], plot["start"])]
    plot["lbl"] = plot["model"].str.replace("-Base", "", regex=False) + "/" + plot["start"]
    plot = plot.dropna(subset=["sft"]).set_index("lbl")[["sft", "lora"]]
    plot.plot(kind="bar", ax=ax[0], color=[C["sft"], C["lora"]])
    ax[0].set_ylabel("juiz 0-5"); ax[0].set_title("Q3 - SFT pleno vs LoRA (recall)")
    ax[0].tick_params(axis="x", rotation=90)
    delta = (plot["lora"] - plot["sft"])
    ax[1].bar(range(len(delta)), delta.values,
              color=[C["good"] if v >= 0 else C["bad"] for v in delta.values])
    ax[1].axhline(0, color="black", lw=0.8); ax[1].set_xticks(range(len(delta)))
    ax[1].set_xticklabels(delta.index, rotation=90, ha="center")
    ax[1].set_ylabel("delta juiz (LoRA - SFT)")
    ax[1].set_title("Q3 - vantagem do LoRA (treina ~1.7% dos params)")
    plt.tight_layout(); plt.show()
    print("média delta (LoRA - SFT pleno):", round(delta.mean(), 3),
          "| LoRA >= SFT em", int((delta >= 0).sum()), "de", len(delta))
'''
)
md("### 2.4 Perplexidade da resposta - base vs SFT vs LoRA")
code(
    '''
# perplexidade da resposta (antes/depois): o antes/depois mais limpo da Q2/Q3 (menor melhor)
sft = load("q2_sft.csv"); lora = load("q3_lora.csv")
if sft is None or lora is None: skip("q2_sft.csv / q3_lora.csv")
else:
    def norm_model(x): return x.replace("-Base", "").replace("-pt", "")
    s = sft[sft.eval_set == "recall"].copy(); s["k"] = s["model"].map(norm_model)
    l = lora[(lora.eval_set == "recall") & (lora.start == "base")].copy(); l["k"] = l["model"].map(norm_model)
    base = s[s.condition == "antes"].groupby("k")["mean_resp_ppl"].first()
    sftp = s[(s.condition == "sft") & (s.start == "base")].groupby("k")["mean_resp_ppl"].first()
    lo = l.groupby("k")["mean_resp_ppl"].first()
    piv = pd.DataFrame({"base": base, "SFT pleno": sftp, "LoRA": lo}).dropna(how="all")
    piv = piv.loc[piv.index.intersection(["Qwen3-0.6B", "Qwen3-1.7B", "gemma-3-1b"])]
    if len(piv):
        ax = piv.plot(kind="bar", color=[C["base"], C["sft"], C["lora"]], figsize=(9, 4))
        ax.set_ylabel("perplexidade da resposta (menor melhor)")
        ax.set_title("Q2/Q3 - perplexidade da resposta: base vs SFT vs LoRA")
        ax.tick_params(axis="x", rotation=0); bar_labels(ax); plt.tight_layout(); plt.show()
    else:
        skip("modelos comuns para perplexidade da resposta")
'''
)
md("### 2.5 Saturacao - qualidade vs rank do LoRA e vs volume de dados de SFT")
code(
    '''
rk = load("q3_rank_sweep.csv"); dc = load("q2_data_curve.csv")
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
if rk is not None:
    rk = rk.copy(); rk["r"] = rk["model"].str.extract(r"r(\\d+)").astype(float)
    rk = rk.dropna(subset=["r"]).sort_values("r")
    ax[0].plot(rk["r"], rk["mean_judge"], "o-", color=C["lora"])
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("rank LoRA")
    ax[0].set_ylabel("juiz 0-5"); ax[0].set_title("Q3 - qualidade vs rank")
else: ax[0].set_title("q3_rank_sweep.csv ausente")
if dc is not None:
    dc = dc.copy(); dc["n"] = dc["model"].str.extract(r"n(\\d+)").astype(float)
    dc = dc.dropna(subset=["n"]).sort_values("n")
    ax[1].plot(dc["n"], dc["mean_judge"], "o-", color=C["sft"])
    ax[1].set_xlabel("número de pares de treino"); ax[1].set_ylabel("juiz 0-5")
    ax[1].set_title("Q2 - qualidade vs volume de dados")
else: ax[1].set_title("q2_data_curve.csv ausente")
plt.tight_layout(); plt.show()
'''
)

# ================================================================ Q4
md(
    """
## 3. Q4 - Destilação de conhecimento (professor -> aluno)

Destilação transfere a capacidade de um modelo grande (**professor**) para um menor
(**aluno**). O professor gera um dataset sintético de perguntas e respostas ancoradas nos
diários, e o aluno é treinado nesses dados. A avaliação usa um benchmark de 100 perguntas
(recall in-domain), com juiz fixo e perplexidade da resposta.

- **Transfer ratio** = (aluno destilado - aluno base) / (professor - aluno base): a fração
  do buraco entre o aluno e o professor que o treino fechou. 1,0 significa que o aluno
  alcançou o professor; 0 significa que não saiu do lugar.
- **response-based vs logit-KD:** treinar nas respostas do professor (texto) ou alinhar as
  distribuições de saída (logits). Compara-se também quatro professores diferentes, com
  orçamento fixo de pares de treino, para ver se a escolha do professor importa.
"""
)
md("### 3.1 base vs distilado (juiz) e transfer ratio por aluno")
code(
    '''
df = load("q4_distill.csv")
if df is None: skip("q4_distill.csv")
else:
    d = df.copy().sort_values("base_judge")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    idx = np.arange(len(d)); w = 0.38
    ax[0].bar(idx - w/2, d["base_judge"], w, label="base", color=C["base"])
    ax[0].bar(idx + w/2, d["distill_judge"], w, label="distilado", color=C["distill"])
    ax[0].axhline(d["teacher_judge"].iloc[0], ls="--", c="black", lw=1, label="teacher ref")
    ax[0].set_xticks(idx)
    ax[0].set_xticklabels(d["student"] + " (" + d["params"] + ")", rotation=90, fontsize=8)
    ax[0].set_ylabel("juiz 0-5"); ax[0].set_title("Q4 - base vs distilado por aluno"); ax[0].legend()
    tr = d.dropna(subset=["transfer_ratio"]).copy()
    tr = tr[tr["transfer_ratio"].between(-0.01, 1.5)]  # esconde caso degenerado (gap negativo)
    ax[1].bar(tr["student"], tr["transfer_ratio"],
              color=[C["good"] if v > 0 else C["bad"] for v in tr["transfer_ratio"]])
    ax[1].set_ylabel("transfer ratio (fração do gap fechado)")
    ax[1].set_title("Q4 - transferência por aluno"); ax[1].tick_params(axis="x", rotation=90)
    plt.tight_layout(); plt.show()
'''
)
md("### 3.2 Perplexidade da resposta - base vs distilado")
code(
    '''
# perplexidade da resposta despenca em todos os alunos (sinal central da destilacao)
df = load("q4_distill.csv")
if df is None: skip("q4_distill.csv")
else:
    d = df.dropna(subset=["base_ppl", "distill_ppl"]).copy()
    d["lbl"] = d["student"] + " (" + d["params"] + ")"
    d = d.sort_values("base_ppl")
    idx = np.arange(len(d)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(idx - w/2, d["base_ppl"], w, label="base", color=C["base"])
    ax.bar(idx + w/2, d["distill_ppl"], w, label="distilado", color=C["distill"])
    ax.set_yscale("log")  # GPT-2 base ~1500 achataria as demais barras
    ax.set_xticks(idx); ax.set_xticklabels(d["lbl"], rotation=90, fontsize=8)
    ax.set_ylabel("perplexidade da resposta (menor melhor, log)")
    ax.set_title("Q4 - perplexidade da resposta: base vs distilado")
    ax.legend(); plt.tight_layout(); plt.show()
'''
)
md("### 3.3 O professor importa? Heatmap (professor x aluno) e ranking medio")
code(
    '''
tc = load("q4_teacher_compare.csv")
if tc is None: skip("q4_teacher_compare.csv")
else:
    piv = tc.pivot_table(index="student", columns="teacher", values="mean_judge", aggfunc="first")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})
    im = ax[0].imshow(piv.values, cmap="RdYlGn", aspect="auto")
    ax[0].set_xticks(range(len(piv.columns))); ax[0].set_xticklabels(piv.columns, rotation=20, ha="right")
    ax[0].set_yticks(range(len(piv.index))); ax[0].set_yticklabels(piv.index, fontsize=8)
    annot_heat(ax[0], piv, im)
    ax[0].set_title("Q4 - juiz por (professor x aluno)"); fig.colorbar(im, ax=ax[0], fraction=0.046)
    means = tc.groupby("teacher")["mean_judge"].mean().sort_values()
    ax[1].barh(means.index, means.values, color="teal")
    ax[1].set_xlabel("juiz médio (alunos)"); ax[1].set_title("Q4 - ranking de professor")
    for y, v in enumerate(means.values): ax[1].text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout(); plt.show()
'''
)
md("### 3.4 Metodos de destilacao e transferencia vs tamanho do aluno")
code(
    '''
me = load("q4_methods.csv"); ds = load("q4_distill.csv")
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
if me is not None:
    m = me.set_index("method")["mean_judge"]
    ax[0].bar(m.index, m.values, color=[C["base"], C["distill"], C["lora"], C["depois"]][:len(m)])
    ax[0].set_ylabel("juiz 0-5"); ax[0].set_title("Q4 - métodos (mesmo aluno Qwen3-0.6B)")
    ax[0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(m.values): ax[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
else: ax[0].set_title("q4_methods.csv ausente")
if ds is not None:
    d = ds.copy(); d["m"] = d["params"].map(params_to_m)
    d = d.dropna(subset=["m", "transfer_ratio"]); d = d[d["transfer_ratio"].between(-0.01, 1.5)]
    ax[1].scatter(d["m"], d["transfer_ratio"], s=60, color=C["distill"])
    for _, r in d.iterrows():
        ax[1].annotate(r["student"], (r["m"], r["transfer_ratio"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax[1].set_xscale("log"); ax[1].set_xlabel("parâmetros do aluno (M, log)")
    ax[1].set_ylabel("transfer ratio"); ax[1].set_title("Q4 - transferência vs tamanho")
    if len(d) > 2:
        c = np.corrcoef(np.log10(d["m"]), d["transfer_ratio"])[0, 1]
        ax[1].text(0.05, 0.92, f"corr(log size, transfer) = {c:.2f}", transform=ax[1].transAxes, fontsize=8)
else: ax[1].set_title("q4_distill.csv ausente")
plt.tight_layout(); plt.show()
'''
)

# ================================================================ Q5
md(
    """
## 4. Q5 - RAG (geração aumentada por recuperação)

RAG busca trechos relevantes num índice e os entrega ao modelo antes de gerar, ancorando a
resposta em evidência recuperada. Modos avaliados:

- **baseline:** sem recuperação (closed-book), o modelo responde só com o que sabe.
- **standard:** recupera os trechos mais similares e gera.
- **agentic_vector:** um agente itera sobre o índice vetorial.
- **agentic_graph:** um agente usa também um grafo de entidades, útil em perguntas que
  cruzam fontes (multi-hop).

O benchmark tem 30 perguntas (factual e multi-hop), pontuadas pelo juiz fixo Qwen3-8B para
comparar motores de forma justa. O **hit-rate@k** isola o retriever: mede se a resposta
esperada aparece em algum dos k trechos recuperados, ou seja, o teto do que o gerador pode
acertar (se a evidência não chega ao prompt, o gerador não tem como acertar).
"""
)
md("### 4.1 Leaderboard motor x modo (juiz fixo 8B)")
code(
    '''
e = load("q5_engines.csv")
if e is None: skip("q5_engines.csv")
else:
    e = e.copy(); e["lbl"] = e["engine"] + " (" + e["params"] + ", " + e["kind"] + ")"
    piv = e.pivot_table(index="lbl", columns="mode", values="judge", aggfunc="first")
    # baseline/standard/agentic_graph sao medidos em quase todos os motores; um vao vazio
    # nessas colunas indica falha real (OOM), nao um modo nao avaliado.
    modes = [m for m in ["baseline", "standard", "agentic_graph"] if m in piv.columns]
    piv = piv[modes].sort_values("standard", na_position="first")
    ax = piv.plot(kind="barh", figsize=(9.5, 7.5))
    ax.set_xlabel("juiz 0-5 (maior melhor)")
    ax.set_title("Q5 - motor x modo (juiz fixo Qwen3-8B; vão vazio = OOM)")
    ax.axvline(2.70, ls="--", c="gray", lw=1); ax.text(2.72, 0.1, "8B standard 2.70", color="gray", fontsize=8)
    for cont, mode in zip(ax.containers, piv.columns):
        for patch, val in zip(cont.patches, piv[mode].values):
            y = patch.get_y() + patch.get_height() / 2
            if pd.isna(val):
                ax.text(0.04, y, "OOM", va="center", ha="left", fontsize=6.5, color=C["bad"], fontweight="bold")
            elif val == 0:
                ax.text(0.04, y, "0", va="center", ha="left", fontsize=6.5, color=C["neutral"])
    ax.legend(title="modo", loc="lower right")
    plt.tight_layout(); plt.show()
'''
)
md("### 4.2 Distribuicao das notas por motor (modo standard)")
code(
    '''
engine_files = {
    "Qwen3-8B (8B)": "benchmark_rag_compare_qwen8b.csv",
    "gemma-3-27b-it (27B)": "q5_engine_gemma-3-27b-it.csv",
    "Qwen3-30B": "q5_rag_30b.csv",
    "gemma-3-1b-it (1B)": "benchmark_rag_compare_gemma1b_it.csv",
    "gemma-3-1b-pt (1B)": "benchmark_rag_compare_gemma1b_pt.csv",
    "qwen2.5-0.5b-distill": "q5_student_qwen2.5-0.5b-distill.csv",
    "qwen3-0.6b-distill": "q5_student_qwen3-0.6b-distill.csv",
    "gemma-1b-distill": "q5_student_gemma-1b-distill.csv",
    "smollm2-360m-distill": "q5_student_smollm2-360m-distill.csv",
    "smollm2-135m-distill": "q5_student_smollm2-135m-distill.csv",
}
data, labels = [], []
for name, f in engine_files.items():
    r = load_rag_scores(f)
    if r is not None and "standard" in r.columns:
        s = r["standard"].dropna()
        if len(s): data.append(s.values); labels.append(name)
if not data:
    skip("arquivos por-pergunta do Q5")
else:
    order = np.argsort([np.median(d) for d in data])
    data = [data[i] for i in order]; labels = [labels[i] for i in order]
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(1, len(data) + 1)
    bp = ax.boxplot(data, vert=False, patch_artist=True, showmeans=True, positions=positions,
                    flierprops=dict(marker="", markersize=0))
    for patch in bp["boxes"]: patch.set_facecolor(C["rag"]); patch.set_alpha(0.45)
    rng = np.random.default_rng(0)
    for pos, vals in zip(positions, data):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(vals, pos + jitter, s=18, color="#08415c", alpha=0.4, zorder=3, edgecolors="none")
    ax.set_yticks(positions); ax.set_yticklabels(labels)
    ax.set_xlabel("juiz 0-5 por pergunta (modo standard)")
    ax.set_title("Q5 - distribuição das notas por motor (caixa + pontos reais)")
    plt.tight_layout(); plt.show()
'''
)
md("### 4.3 Contribuicao do RAG vs baseline (8B, pareado por pergunta)")
code(
    '''
r = load_rag_scores("benchmark_rag_compare_qwen8b.csv", modes=("baseline", "standard", "agentic_graph"))
if r is None: skip("benchmark_rag_compare_qwen8b.csv")
else:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    rr = r.dropna(subset=["baseline", "standard"])
    delta = rr["standard"] - rr["baseline"]
    win = int((delta > 0).sum()); tie = int((delta == 0).sum()); loss = int((delta < 0).sum())
    ax[0].bar(["RAG vence", "empate", "RAG perde"], [win, tie, loss],
              color=[C["good"], C["neutral"], C["bad"]])
    ax[0].set_ylabel("perguntas"); ax[0].set_title(f"Q5 - standard vs baseline (n={len(rr)})")
    for i, v in enumerate([win, tie, loss]): ax[0].text(i, v, str(v), ha="center", va="bottom")
    ax[1].hist(delta, bins=np.arange(-5, 6) - 0.5, color=C["rag"], edgecolor="white")
    ax[1].axvline(0, color="red", ls="--", lw=1.4, label="sem efeito (0)")
    ax[1].axvline(delta.mean(), ls="--", c="black", lw=1, label=f"média {delta.mean():+.2f}")
    ax[1].set_xlabel("vantagem do RAG em pontos do juiz (standard - baseline)")
    ax[1].set_ylabel("número de perguntas")
    ax[1].set_title("Q5 - distribuição da vantagem/desvantagem de usar RAG (em pontos)")
    ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.show()
'''
)
md("### 4.4 Ganho do RAG por tipo de pergunta (factual vs multi-hop)")
code(
    '''
r = load_rag_scores("benchmark_rag_compare_qwen8b.csv", modes=("baseline", "standard", "agentic_graph"))
if r is None or "type" not in r.columns: skip("type em benchmark_rag_compare_qwen8b.csv")
else:
    g = r.groupby("type")[["baseline", "standard", "agentic_graph"]].mean()
    ax = g.plot(kind="bar", figsize=(9, 4), color=[C["base"], C["sft"], C["rag"]])
    ax.set_ylabel("juiz 0-5 médio"); ax.set_title("Q5 - desempenho por tipo de pergunta (motor 8B)")
    ax.tick_params(axis="x", rotation=0); bar_labels(ax); plt.tight_layout(); plt.show()
'''
)
md("### 4.5 Teto do retriever (hit-rate@k) e tamanho do motor vs qualidade")
code(
    '''
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ret = load("q5_retrieval.csv")
if ret is not None:
    ks = [1, 3, 5, 10]
    for method in ret["method"].unique():
        row = ret[ret["method"] == method]
        if len(row): ax[0].plot(ks, [row[f"hit@{k}"].iloc[0] for k in ks], "o-", label=method)
    ax[0].set_xlabel("k"); ax[0].set_ylabel("hit-rate@k"); ax[0].set_ylim(0, 1.02)
    ax[0].set_title("Q5 - retrieval hit-rate@k (teto do RAG)"); ax[0].legend()
else: ax[0].set_title("q5_retrieval.csv ausente")
e = load("q5_engines.csv")
if e is not None:
    d = e[e["mode"] == "standard"].copy(); d["m"] = d["params"].map(params_to_m)
    d = d.dropna(subset=["m"])
    for kind, grp in d.groupby("kind"):
        ax[1].scatter(grp["m"], grp["judge"], s=55, label=kind)
    ax[1].set_xscale("log"); ax[1].set_xlabel("parâmetros do motor (M, log)")
    ax[1].set_ylabel("juiz standard"); ax[1].set_title("Q5 - tamanho do motor vs qualidade")
    ax[1].legend(fontsize=8)
else: ax[1].set_title("q5_engines.csv ausente")
plt.tight_layout(); plt.show()
'''
)
md("### 4.6 Estrategias de indice (licitacoes): media do modo standard")
code(
    '''
strat = {
    "full": "strategy_full.csv", "full+mmr": "strategy_full_mmr.csv",
    "dedup": "strategy_dedup.csv", "dedup+mmr": "strategy_dedup_mmr.csv",
    "balanced->fullbench": "strategy_balanced_on_fullbench.csv",
    "targeted full": "targeted_full.csv", "targeted full+mmr": "targeted_full_mmr.csv",
    "targeted nonlic": "targeted_nonlic_only.csv",
}
vals = {}
for name, f in strat.items():
    r = load_rag_scores(f, modes=("baseline", "standard"))
    if r is not None and "standard" in r.columns:
        vals[name] = r["standard"].dropna().mean()
if not vals: skip("strategy_*.csv / targeted_*.csv")
else:
    s = pd.Series(vals).sort_values()
    ax = s.plot(kind="barh", figsize=(9, 4), color=C["rag"])
    ax.set_xlabel("juiz standard médio"); ax.set_title("Q5 - estratégias de índice/licitações")
    for y, v in enumerate(s.values): ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8)
    plt.tight_layout(); plt.show()
'''
)

# ================================================================ Q6
md(
    """
## 5. Q6 - Guardrails (camada de proteção)

Guardrails são a camada de controle que filtra entrada e saída do modelo: bloqueia
jailbreak e pedidos inseguros e mascara dados pessoais (PII) como CPF, CNPJ, CEP, telefone
e email. O benchmark tem 30 perguntas (10 adversariais, 5 com PII na saída, 15 benignas) e
mede a taxa tratada **com** e **sem** a camada. O ponto central é o dilema *helpfulness vs
harmlessness*: bloquear o que é nocivo sem barrar os pedidos legítimos (sem falsos
positivos nas benignas). Um teste extra com ataques **parafraseados** mostra a fragilidade
de filtros por regex a reformulações.
"""
)
code(
    '''
df = load("q6_guardrails.csv"); adv = load("q6_adversarial.csv")
if df is None: skip("q6_guardrails.csv")
else:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    df.plot(x="type", y=["rate_without", "rate_with"], kind="bar", ax=ax[0], color=[C["bad"], C["good"]])
    ax[0].set_title("Q6 - taxa tratada (benchmark padrão)"); ax[0].set_ylim(0, 1.08)
    ax[0].yaxis.set_major_formatter(PercentFormatter(1.0)); ax[0].tick_params(axis="x", rotation=15)
    ax[0].legend(["sem guardrails", "com guardrails"])
    if adv is not None:
        adv.plot(x="type", y=["rate_with"], kind="bar", ax=ax[1], color=C["bad"], legend=False)
        ax[1].set_title("Q6 - ataques PARAFRASEADOS (regex evade)"); ax[1].set_ylim(0, 1.08)
        ax[1].yaxis.set_major_formatter(PercentFormatter(1.0)); ax[1].tick_params(axis="x", rotation=15)
    plt.tight_layout(); plt.show(); print(df.to_string(index=False))
'''
)

# ================================================================ synthesis
md(
    """
## 6. Síntese transversal

Leituras que só aparecem ao cruzar as questões: um mapa de melhor desempenho por família em
cada tarefa de pós-treino e RAG, e a relação entre tamanho do motor e qualidade do RAG. As
escalas de juiz são comparáveis (Q2/Q3/Q4 no recall dos docentes; Q5 no RAG).
"""
)
code(
    '''
# melhor juiz por familia em cada questao (Q2/Q3/Q4 docentes-recall; Q5 RAG)
panel = {}
sft = load("q2_sft.csv"); lora = load("q3_lora.csv"); dist = load("q4_distill.csv"); eng = load("q5_engines.csv")
def fam_of(name):
    n = name.lower()
    if "qwen2.5" in n or "qwen2p5" in n: return "qwen2.5"
    if "qwen3" in n or "qwen" in n: return "qwen3"
    if "gemma" in n: return "gemma"
    if "smollm" in n: return "smollm2"
    if "gpt2" in n: return "gpt2"
    return n
rows = []
if sft is not None:
    d = sft[(sft.eval_set == "recall") & (sft.condition == "sft")]
    for _, r in d.iterrows(): rows.append((fam_of(r["model"]), "Q2 SFT", r["mean_judge"]))
if lora is not None:
    d = lora[lora.eval_set == "recall"]
    for _, r in d.iterrows(): rows.append((fam_of(r["model"]), "Q3 LoRA", r["mean_judge"]))
if dist is not None:
    for _, r in dist.iterrows(): rows.append((fam_of(r["student"]), "Q4 distill", r["distill_judge"]))
if eng is not None:
    d = eng[eng["mode"] == "standard"]
    for _, r in d.iterrows(): rows.append((fam_of(r["engine"]), "Q5 RAG", r["judge"]))
if not rows:
    skip("CSVs de Q2-Q5")
else:
    M = pd.DataFrame(rows, columns=["family", "task", "judge"]).pivot_table(
        index="family", columns="task", values="judge", aggfunc="max")
    M = M.reindex(columns=[c for c in ["Q2 SFT", "Q3 LoRA", "Q4 distill", "Q5 RAG"] if c in M.columns])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(M.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=4)
    ax.set_xticks(range(len(M.columns))); ax.set_xticklabels(M.columns)
    ax.set_yticks(range(len(M.index))); ax.set_yticklabels(M.index)
    annot_heat(ax, M, im)
    ax.set_title("Melhor juiz por família x tarefa (maior melhor)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.show()
'''
)
code(
    '''
# escala global: parametros vs melhor juiz standard de RAG (motores) com rotulos
e = load("q5_engines.csv")
if e is None: skip("q5_engines.csv")
else:
    d = e[e["mode"] == "standard"].copy(); d["m"] = d["params"].map(params_to_m)
    d = d.dropna(subset=["m"]).sort_values("m")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    markers = {"instruct": "o", "base": "s", "distill-student": "^", "instruct-4bit": "D"}
    for kind, grp in d.groupby("kind"):
        ax.scatter(grp["m"], grp["judge"], s=80, marker=markers.get(kind, "o"), label=kind, zorder=3)
    ax.axhline(2.70, ls="--", c="gray", lw=1); ax.text(d["m"].min(), 2.74, "8B standard 2.70", color="gray", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("parâmetros (M, escala log)"); ax.set_ylabel("juiz standard (RAG)")
    ax.set_title("Síntese - tamanho do motor vs qualidade do RAG (aluno destilado de 0.5B lidera)")
    ax.legend(fontsize=8, title="tipo")
    texts = [ax.text(r["m"], r["judge"], r["engine"], fontsize=7) for _, r in d.iterrows()]
    if HAS_ADJUST:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5), expand=(1.4, 1.8))
    plt.tight_layout(); plt.show()
'''
)

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": ".venv (llm-finetuning)", "language": "python", "name": "llmft"}
nb["metadata"]["language_info"] = {"name": "python"}
with open("notebooks/graficos_resultados.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote notebook with", len(cells), "cells")
