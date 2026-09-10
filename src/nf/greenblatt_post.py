"""1x2 post figure: Astra (prompted no-CoT) on AIME-Plus-Plus (post-cutoff, AIME tier + all tiers) and Gen-Arithmetic 15 ops."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import aimepp_tier, load, tag_dir

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
ARMS = {
    "CB": ("counting filler 1..N", "#6ACC65", "o", "-"),
    "XB": ("300 dots", "#4878CF", "D", "none"),
    "RB": ("question repeats", "#D65F5F", "s", "--"),
}
PANELS = {("aimepp", "AIME"): "AIME-Plus-Plus (Aug 2026), AIME tier", ("arith", (15,)): "Gen-Arithmetic, 15 ops"}
MODEL = "gpt-6-astra:low"


def main():
    df = pd.concat([load("greenblatt"), load("greenblatt2")]).reset_index(drop=True)
    df = df[(df.status == "completed") & (df.model_eff == MODEL) & df.arm.isin(["B", *ARMS])]
    df = df[((df.arm != "B") | (df.k == 0)) & (df.reasoning_tokens.fillna(0) == 0)]
    df["x"] = np.where(df.arm == "B", 0, df.filler_tokens.fillna(df.k))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, ((task, depths), title) in zip(axes, PANELS.items()):
        d = df[df.task == task]
        if task == "aimepp":
            d = d[d.problem_id.map(aimepp_tier) == depths]
        elif depths:
            d = d[d.depth.isin(depths)]
        b = d[d.arm == "B"].correct.mean()
        ax.axhline(b, color="gray", ls=":", lw=1.2, label=f"no filler ({b:.2f})")
        for arm, (lab, col, mk, ls) in ARMS.items():
            g = (
                d[d.arm == arm]
                .groupby("k")
                .agg(x=("x", "mean"), acc=("correct", "mean"), n=("correct", "size"))
                .sort_values("x")
            )
            if g.empty:
                continue
            ax.errorbar(
                g.x,
                g.acc,
                yerr=1.96 * np.sqrt(g.acc * (1 - g.acc) / g.n),
                fmt=mk,
                ls=ls,
                color=col,
                lw=1.6,
                capsize=2,
                ms=5,
                label=lab,
            )
        ax.set_xscale("symlog", linthresh=10)
        ax.set(xlim=(-1, 4000), ylim=(-0.02, 1.02), title=title, xlabel="filler tokens (symlog)")
        ax.legend(frameon=False, fontsize=8, loc="lower left" if task == "aimepp" else "upper left")
    axes[0].set_ylabel("accuracy, no CoT")
    fig.suptitle(f"{MODEL} (prompted no-CoT): accuracy vs user-supplied filler", fontsize=12)
    out = tag_dir("greenblatt") / "figs" / "greenblatt_post_1x2.png"
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
