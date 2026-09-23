"""Few-shot elicitation check for gpt-6-astra: does the filler dose-response survive when the no-CoT format is fully
demonstrated? Compares the paper's zero-shot runs with 3-shot runs (three gold-answer demonstrations from held-out
problems, each carrying the same dot count as the query; `nf.run --shots 3`, tag `fewshot_astra`; 10-shot: tag
`fewshot10_astra`, N-hop query set cut to 140) on four tasks.
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
    few10 = _valid(load("fewshot10_astra"))
    zero = {t: _valid(load(t)) for t in {p[2] for p in PANELS.values()}}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    rows = []
    for ax, (key, (task, depth, ztag, title)) in zip(axes.flat, PANELS.items()):
        for label, d, col, mk in (
            ("0-shot (paper)", _select(zero[ztag], task, depth), "tab:gray", "o"),
            ("3-shot", _select(few, task, depth), "tab:red", "s"),
            ("10-shot", _select(few10, task, depth), "tab:blue", "^"),
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
                        "shots": int(label.split("-")[0]),
                        "k": int(r.k),
                        "acc": round(r.acc, 4),
                        "n": int(r.n),
                    }
                )
        ax.set_xscale("symlog", linthresh=4)
        ax.set(xlim=(-0.5, MAX_K * 1.6), ylim=(-0.02, 1.02), title=title, xlabel="filler tokens", ylabel="accuracy")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("gpt-6-astra, no-CoT: zero-shot vs 3- and 10-shot", fontsize=12)
    out = tag_dir("fewshot_astra") / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fewshot_astra.png", dpi=160)
    pd.DataFrame(rows).to_csv(out / "fewshot_astra.csv", index=False)
    print(pd.DataFrame(rows).pivot_table(index=["panel", "shots"], columns="k", values="acc").round(2).to_string())


OTHERS = {"gpt-5.6-sol": "Sol", "claude-opus-4-5-20251101": "Opus 4.5", "deepseek/deepseek-v3.2": "DeepSeek V3.2"}


def _valid_model(df, model, max_k):
    df = df[(df.model == model) & (df.status == "completed")]
    df = df[(df.provider == "anthropic") | (df.reasoning_tokens.fillna(0) == 0)]
    return df[((df.arm == "B") & (df.k == 0)) | ((df.arm == "XB") & (df.k <= max_k))]


def others():
    """Sol / Opus 4.5 / DeepSeek V3.2: zero-shot (paper runs, dots ≤ 1,024) vs 10-shot (`fewshot10_others`)."""
    few = load("fewshot10_others")
    zero = {t: load(t) for t in {p[2] for p in PANELS.values()}}
    fig, axes = plt.subplots(3, 4, figsize=(15, 9.5), constrained_layout=True)
    rows = []
    for i, (model, mlab) in enumerate(OTHERS.items()):
        for j, (key, (task, depth, ztag, title)) in enumerate(PANELS.items()):
            if task == "nhop":  # the other models only have signal at 2 hops (the post's mixed-hop comparison)
                depth, title = 2, "N-hop natural facts, 2 hops"
            ax = axes[i, j]
            for label, d, col, mk in (
                ("0-shot (paper)", _select(_valid_model(zero[ztag], model, 1024), task, depth), "tab:gray", "o"),
                ("10-shot", _select(_valid_model(few, model, 1024), task, depth), "tab:blue", "^"),
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
                            "model": mlab,
                            "panel": key,
                            "shots": int(label.split("-")[0]),
                            "k": int(r.k),
                            "acc": round(r.acc, 4),
                            "n": int(r.n),
                        }
                    )
            ax.set_xscale("symlog", linthresh=4)
            ax.set(
                xlim=(-0.5, 1024 * 1.6),
                ylim=(-0.02, 1.02),
                title=f"{mlab}: {title}",
                xlabel="filler tokens",
                ylabel="accuracy",
            )
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Other models, no-CoT: zero-shot vs 10-shot", fontsize=12)
    out = tag_dir("fewshot10_others") / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fewshot10_others.png", dpi=160)
    pd.DataFrame(rows).to_csv(out / "fewshot10_others.csv", index=False)


if __name__ == "__main__":
    import sys

    others() if "others" in sys.argv else main()
