"""Dose-response in the number of filler dots (arms XB = user-supplied, XC = model-emitted; k = exact dot count).
usage: uv run -m nf.dots_dose dots_dose [more tags]"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import ROOT, load
from .prompts import build
from .tasks import make_problem


def to_dot_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Relabel token-matched dot arms (DB/DC, k = copies-equivalent) as exact-count arms (XB/XC, k = number of dots),
    using the mean dot count of that (depth, k) cell so points line up on a shared axis."""
    df = df.copy()
    m = df.arm.isin(["DB", "DC"])
    if m.any():
        cache = {}

        def dots(row):
            key = (row.depth, row.k)
            if key not in cache:
                cache[key] = build(make_problem(row.depth, 0), "DC", row.k).expected_prefix.count(".")
            return cache[key]

        df.loc[m, "k"] = df[m].apply(dots, axis=1)
        df.loc[m, "arm"] = df.loc[m, "arm"].map({"DB": "XB", "DC": "XC"})
    return df


matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 12})


def se(p, n):
    p, n = np.asarray(p, float), np.asarray(n, float)
    return 1.96 * np.sqrt(p * (1 - p) / n)


def table(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["model_eff", "depth", "arm", "k"])
        .agg(
            n=("correct", "size"),
            acc=("correct", "mean"),
            compliant=("compliant", "mean"),
            rt=("reasoning_tokens", "mean"),
        )
        .reset_index()
    )
    return g


def figure(t: pd.DataFrame, out):
    models = sorted(t.model_eff.unique())
    fig, axes = plt.subplots(
        1, len(models), figsize=(6.5 * len(models), 5.5), constrained_layout=True, squeeze=False, sharey=True
    )
    cmap = plt.get_cmap("viridis")
    for ax, me in zip(axes[0], models):
        d = t[t.model_eff == me]
        depths = sorted(d.depth.unique())
        for i, depth in enumerate(depths):
            col = cmap(i / max(1, len(depths) - 1))
            base = d[(d.depth == depth) & (d.k == 0)]
            if len(base):
                ax.axhline(base.acc.iloc[0], color=col, ls=":", lw=1)
            for arm, ls, mk in (("XB", "-", "o"), ("XC", "--", "s")):
                s = d[(d.depth == depth) & (d.arm == arm)].sort_values("k")
                if s.empty:
                    continue
                ax.errorbar(
                    s.k,
                    s.acc,
                    yerr=se(s.acc, s.n),
                    fmt=mk,
                    ls=ls,
                    color=col,
                    lw=2,
                    capsize=3,
                    ms=5,
                    label=f"depth {depth}, {'user-supplied' if arm == 'XB' else 'model-emitted'} dots",
                )
        ax.set_xscale("log")
        ticks = [v for v in (1, 2, 5, 10, 20, 40, 70, 140, 340, 700, 1400, 2800) if v <= d.k.max() * 1.05]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(v) for v in ticks])
        ax.axhline(0.125, color="gray", lw=1, ls="-.")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(me, fontsize=13)
        ax.set_xlabel("number of filler dots (tokens), log axis", fontsize=12)
        ax.legend(frameon=False, fontsize=8, ncol=1, loc="lower right")
    axes[0, 0].set_ylabel(
        "accuracy, permutation task, no CoT (↑)\n(dotted = same depth at 0 dots; dash-dot = chance)", fontsize=11
    )
    fig.suptitle("Dose-response in filler dots: accuracy vs number of dots, 1 to ~4000 (n=40 per point)", fontsize=14)
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.concat([load(t) for t in sys.argv[1:]])
    df = df[(df.status != "error") & (df.task == "perm") & df.arm.isin(["B", "XB", "XC", "DB", "DC"])]
    df = df[(df.arm != "B") | (df.k == 0)]
    df = to_dot_counts(df)
    out = ROOT / "results" / sys.argv[1] / "figs"
    out.mkdir(exist_ok=True)
    t = table(df)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    piv = t.pivot_table(index=["model_eff", "depth", "arm"], columns="k", values="acc")
    print(piv.round(2).to_string())
    print(
        "\ncompliance (model-emitted arm):",
        t[t.arm == "XC"].compliant.mean().round(3),
        "| max reasoning tokens:",
        df.reasoning_tokens.max(),
    )
    t.to_csv(out / "dots_dose.csv", index=False)
    figure(t, out / "dots_dose.png")
    print("fig ->", out / "dots_dose.png")
