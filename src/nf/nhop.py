"""Natural-fact n-hop (Greenblatt multi_hop extended to 2-7 hops): accuracy vs hops per filler condition.

Left panel: accuracy vs hops, one line per filler condition. Right panel: accuracy vs filler tokens (log x), one line
per hop count. Prints a table of paired gains vs the no-filler baseline with exact McNemar p-values."""

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .analyze import ROOT, load

matplotlib.use("Agg")
ARMS = {  # arm -> (label, marker)
    "CB": ("counting filler", "o"),
    "XB": ("dots", "s"),
    "RB": ("repeated question", "^"),
}


def _valid(df, task="nhop"):
    df = df[(df.status != "error") & (df.status != "refusal") & (df.task == task) & df.arm.isin(["B", *ARMS])]
    df = df[(df.arm != "B") | (df.k == 0)]
    return df[df.reasoning_tokens.fillna(0) == 0]


def main(tags=("nhop",), task="nhop"):
    df = _valid(pd.concat([load(t) for t in tags]).reset_index(drop=True), task)
    df["x"] = np.where(df.arm == "B", 0, df.get("filler_tokens", df.k).fillna(df.k)).astype(int)
    tok = df.groupby(["arm", "k"]).x.transform("mean").round().astype(int)  # RB/XB token counts vary per problem
    df["cond"] = np.where(df.arm == "B", "no filler", df.arm + "@" + tok.astype(str))
    models = sorted(df.model_eff.unique())
    models = [m for m in models if m == "gpt-6-astra:low"] or models
    fig, axes = plt.subplots(len(models), 1, figsize=(7.5, 4.6 * len(models)), constrained_layout=True, squeeze=False)
    out = []
    for r, me in enumerate(models):
        d = df[df.model_eff == me]
        acc = d.groupby(["cond", "depth"]).correct.mean().unstack("depth")
        keep = {
            16,
            64,
            256,
            1024,
            4096,
            8192,
            16384,
        }  # dot filler at powers of four; counting/repeat cells stay in the table only
        conds = ["no filler"] + sorted(
            [c for c in acc.index if c.startswith("XB@") and int(c.split("@")[1]) in keep],
            key=lambda c: int(c.split("@")[1]),
        )
        ax = axes[r, 0]
        cmap = plt.get_cmap("viridis")
        for c in conds:
            if c == "no filler":
                ax.plot(acc.columns, acc.loc[c], color="gray", ls=":", marker="x", lw=2, label=c)
                continue
            _arm, x = c.split("@")
            shade = cmap(0.1 + 0.85 * np.log10(int(x) + 1) / np.log10(20000))
            # draw order, bottom to top: 16, 64, 256, 1,024, 16,384, 8,192, 4,096 (4k on top; 8k/16k just under it; 1k under all three)
            z = {16: 3, 64: 4, 256: 5, 1024: 6, 16384: 7, 8192: 8, 4096: 9}.get(int(x), 2)
            ax.plot(
                acc.columns, acc.loc[c], marker="o", color=shade, lw=1.8, label=f"{int(x):,} filler tokens", zorder=z
            )
        ax.set(
            xlabel="hops",
            ylabel="accuracy",
            title="N-hop natural facts: accuracy vs hops (no-CoT Astra)",
            ylim=(-0.02, 1.02),
        )
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
        # paired gains table
        for depth, g in d.groupby("depth"):
            base = g[g.arm == "B"].set_index("problem_id").correct.astype(bool)
            for c in conds[1:]:
                s = g[g.cond == c].set_index("problem_id").correct.astype(bool)
                idx = base.index.intersection(s.index)
                b, t = base[idx], s[idx]
                win, lose = int((t & ~b).sum()), int((b & ~t).sum())
                p = binomtest(win, win + lose, 0.5).pvalue if win + lose else 1.0
                out.append(
                    {
                        "model": me,
                        "hops": depth,
                        "cond": c,
                        "n": len(idx),
                        "base": b.mean(),
                        "filler": t.mean(),
                        "gain": t.mean() - b.mean(),
                        "win": win,
                        "lose": lose,
                        "p": p,
                    }
                )
    res = pd.DataFrame(out)
    pd.set_option("display.width", 200, "display.max_rows", 500)
    print(res.round(3).to_string(index=False))
    res.to_csv(ROOT / "results" / "main" / ("nhop_gains.csv" if task == "nhop" else "nhop_gains_nl.csv"), index=False)
    fig.savefig(ROOT / "results" / "main" / ("nhop_astra.png" if task == "nhop" else "nhop_astra_nl.png"), dpi=130)
    print("wrote results/nhop_astra.png, results/nhop_gains.csv")


if __name__ == "__main__":
    main(task=sys.argv[1] if len(sys.argv) > 1 else "nhop")
