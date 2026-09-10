"""Serial vs parallel arithmetic at fixed op count: chain (height == ops) vs balanced tree (height ~ log2 ops) vs the
original Gen-Arithmetic mix, Astra, no-CoT with and without counting filler. Usage: uv run -m nf.arith_shape"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import load, tag_dir

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
SHAPES = {
    "arithchain": ("chain (height = ops)", "#D65F5F"),
    "arith": ("Gen-Arithmetic mix", "#4878CF"),
    "arithbal": ("balanced (height ~ log2 ops)", "#6ACC65"),
}


def main():
    df = pd.concat([load("greenblatt"), load("arith_shape")]).reset_index(drop=True)
    df = df[
        (df.model_eff == "gpt-6-astra:low")
        & df.task.isin(SHAPES)
        & (df.status == "completed")
        & (df.reasoning_tokens.fillna(0) == 0)
    ]
    df = df[((df.arm == "B") & (df.k == 0)) | ((df.arm == "CB") & df.k.isin([300, 1000]))]
    df = df[df.depth.isin([7, 10, 15])]
    tab = (
        df.assign(cond=np.where(df.arm == "B", "no filler", "count " + df.k.astype(str)))
        .groupby(["task", "depth", "cond"])
        .correct.agg(["mean", "size"])
    )
    pd.set_option("display.width", 200)
    print(tab.round(3).to_string())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True, sharey=True)
    for ax, cond_k in zip(axes, [("B", 0), ("CB", 300), ("CB", 1000)]):
        for task, (lab, col) in SHAPES.items():
            g = (
                df[(df.task == task) & (df.arm == cond_k[0]) & (df.k == cond_k[1])]
                .groupby("depth")
                .correct.agg(["mean", "size"])
            )
            ax.errorbar(
                g.index,
                g["mean"],
                yerr=1.96 * np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"]),
                marker="o",
                color=col,
                capsize=3,
                label=lab,
            )
        ax.set(
            xlabel="operations",
            title="no filler" if cond_k[0] == "B" else f"counting filler 1..{cond_k[1]}",
            ylim=(-0.02, 1.02),
            xticks=[7, 10, 15],
        )
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy, no CoT")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "gpt-6-astra:low (prompted no-CoT): serial (chain) vs parallel (balanced) arithmetic at equal op count"
    )
    out = tag_dir("arith_shape") / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "arith_shape.png", dpi=150)
    tab.to_csv(out / "arith_shape.csv")
    print("wrote", out / "arith_shape.png")


if __name__ == "__main__":
    main()
