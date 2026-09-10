"""Positional control: dots BEFORE the statement (YB) vs AFTER it (XB, user-supplied) vs model-emitted (XC).
usage: uv run -m nf.dots_position dots_before dots_dose fillerD_astra fillerD_sol"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from .analyze import load, tag_dir
from .dots_dose import se, to_dot_counts

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
MODELS = ["gpt-5.6-sol:none", "gpt-6-astra:low"]
STYLE = {
    "YB": ("#B47CC7", "-", "D", "dots BEFORE statement (user)"),
    "XB": ("#4878CF", "-", "o", "dots AFTER statement (user)"),
    "XC": ("#D65F5F", "--", "s", "dots emitted by model"),
}

if __name__ == "__main__":
    df = pd.concat([load(t) for t in sys.argv[1:]])
    df = df[
        (df.status != "error")
        & (df.task == "perm")
        & df.model_eff.isin(MODELS)
        & df.arm.isin(["B", "YB", "XB", "XC", "DB", "DC"])
    ]
    df = df[(df.arm != "B") | (df.k == 0)]
    df = to_dot_counts(df)
    t = df.groupby(["model_eff", "depth", "arm", "k"]).agg(n=("correct", "size"), acc=("correct", "mean")).reset_index()
    depths = {me: sorted(t[(t.model_eff == me) & (t.arm == "YB")].depth.unique()) for me in MODELS}
    ncol = max(len(v) for v in depths.values())
    fig, axes = plt.subplots(
        len(MODELS), ncol, figsize=(6 * ncol, 4.8 * len(MODELS)), constrained_layout=True, squeeze=False, sharey=True
    )
    for r, me in enumerate(MODELS):
        for c, depth in enumerate(depths[me]):
            ax = axes[r, c]
            d = t[(t.model_eff == me) & (t.depth == depth)]
            base = d[d.k == 0]
            if len(base):
                ax.axhline(base.acc.iloc[0], color="gray", ls=":", lw=1, label="no filler")
            for arm, (col, ls, mk, lab) in STYLE.items():
                s = d[(d.arm == arm) & (d.k > 0)].sort_values("k")
                if s.empty:
                    continue
                ax.errorbar(s.k, s.acc, yerr=se(s.acc, s.n), fmt=mk, ls=ls, color=col, lw=2, capsize=3, ms=5, label=lab)
            ax.axhline(0.125, color="gray", lw=1, ls="-.")
            ax.set_xscale("log")
            ticks = [v for v in (1, 5, 20, 70, 140, 340, 700, 1400, 2800) if v <= max(d.k.max(), 1) * 1.05]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(v) for v in ticks])
            ax.set_ylim(-0.03, 1.05)
            ax.set_title(f"{me}, permutation depth {depth}", fontsize=11)
            ax.set_xlabel("number of filler dots (tokens), log")
            if c == 0:
                ax.set_ylabel("accuracy, no CoT (↑)")
            ax.legend(frameon=False, fontsize=8, loc="best")
        for c in range(len(depths[me]), ncol):
            axes[r, c].set_visible(False)
    fig.suptitle(
        "Positional control: the same dots placed before vs after the problem statement (n=40 per point)", fontsize=12
    )
    out = tag_dir("dots_before") / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "dots_before_vs_after.png", dpi=200)
    pd.set_option("display.width", 200)
    print(
        t[t.arm.isin(["YB", "XB"])]
        .pivot_table(index=["model_eff", "depth", "arm"], columns="k", values="acc")
        .round(2)
        .to_string()
    )
    print("fig ->", out / "dots_before_vs_after.png")
