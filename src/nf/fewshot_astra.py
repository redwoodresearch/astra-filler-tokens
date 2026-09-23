"""Few-shot elicitation check for gpt-6-astra: does the filler dose-response survive when the no-CoT format is fully
demonstrated? Compares the paper's zero-shot runs with 3-shot runs (three gold-answer demonstrations from held-out
problems, each carrying the same dot count as the query; `nf.run --shots 3`, tag `fewshot_astra`) on four tasks.
Writes <fewshot_astra>/figs/fewshot_astra.png and fewshot_astra.csv.

  uv run -m nf.fewshot_astra
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import ROOT, aimepp_tier, load

try:
    from .analyze import tag_dir
except ImportError:  # working repo: results/<tag>

    def tag_dir(tag):
        return ROOT / "results" / tag


matplotlib.use("Agg")
MODEL = "gpt-6-astra"
# panel -> (task, depth filter, zero-shot tag, title). AIME++ is restricted to its AIME tier (the only tier with signal).
PANELS = {
    "arith15": ("arith", 15, "greenblatt", "Gen-Arithmetic, 15 ops"),
    "nhop4": ("nhop", 4, "nhop", "N-hop natural facts, 4 hops"),
    "aime": ("aime", None, "greenblatt", "AIME/HMMT (218)"),
    "aimepp": ("aimepp", "AIME", "greenblatt2", "AIME-Plus-Plus, AIME tier (34)"),
}
MAX_K = 4096


def _valid(df):
    df = df[(df.model == MODEL) & (df.status == "completed") & (df.reasoning_tokens.fillna(0) == 0)]
    df = df[((df.arm == "B") & (df.k == 0)) | ((df.arm == "XB") & (df.k <= MAX_K))]
    return df


def _select(df, task, depth):
    d = df[df.task == task]
    if task == "aimepp":
        return d[d.problem_id.map(aimepp_tier) == depth]
    return d[d.depth == depth] if depth else d


def _curve(d):
    d = d.assign(correct=d.correct.astype(bool).astype(float))
    return d.groupby("k").agg(acc=("correct", "mean"), n=("correct", "size")).sort_index().reset_index()


def main():
    few = _valid(load("fewshot_astra"))
    zero = {t: _valid(load(t)) for t in {p[2] for p in PANELS.values()}}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    rows = []
    for ax, (key, (task, depth, ztag, title)) in zip(axes.flat, PANELS.items()):
        for label, d, col, mk in (
            ("0-shot (paper)", _select(zero[ztag], task, depth), "tab:gray", "o"),
            ("3-shot, same-dose gold demos", _select(few, task, depth), "tab:red", "s"),
        ):
            g = _curve(d)
            ax.errorbar(
                g.k,
                g.acc,
                yerr=1.96 * np.sqrt(g.acc * (1 - g.acc) / g.n),
                fmt=mk,
                ls="-",
                color=col,
                lw=1.6,
                capsize=2,
                ms=5,
                label=label,
            )
            for _, r in g.iterrows():
                rows.append(
                    {
                        "panel": key,
                        "shots": 0 if label.startswith("0") else 3,
                        "k": int(r.k),
                        "acc": round(r.acc, 4),
                        "n": int(r.n),
                    }
                )
        ax.set_xscale("symlog", linthresh=4)
        ax.set(xlim=(-0.5, MAX_K * 1.6), ylim=(-0.02, 1.02), title=title, xlabel="filler tokens", ylabel="accuracy")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("gpt-6-astra, no-CoT: zero-shot vs 3-shot (same-dose gold demonstrations)", fontsize=12)
    out = tag_dir("fewshot_astra") / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fewshot_astra.png", dpi=160)
    pd.DataFrame(rows).to_csv(out / "fewshot_astra.csv", index=False)
    print(pd.DataFrame(rows).pivot_table(index=["panel", "shots"], columns="k", values="acc").round(2).to_string())


if __name__ == "__main__":
    main()
