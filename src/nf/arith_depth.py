"""How serial are the Gen-Arithmetic problems? For each expression: number of operations (the generator's `depth`),
tree height (longest chain of nested operations = critical path), and the fraction of ops on that path. Then Astra's
no-CoT accuracy with and without filler as a function of height, within each ops level. Existing data only."""

import ast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import load, tag_dir
from .tasks import make_problem

matplotlib.use("Agg")


def height(node) -> int:
    if isinstance(node, ast.BinOp):
        return 1 + max(height(node.left), height(node.right))
    if isinstance(node, ast.UnaryOp):
        return height(node.operand)
    return 0


def n_ops(node) -> int:
    if isinstance(node, ast.BinOp):
        return 1 + n_ops(node.left) + n_ops(node.right)
    if isinstance(node, ast.UnaryOp):
        return n_ops(node.operand)
    return 0


def main():
    df = pd.concat([load("greenblatt")]).reset_index(drop=True)
    df = df[(df.task == "arith") & (df.status == "completed") & (df.reasoning_tokens.fillna(0) == 0)]
    df = df[((df.arm == "B") & (df.k == 0)) | ((df.arm == "CB") & df.k.isin([300, 1000]))].copy()
    meta = {}
    for pid in df.problem_id.unique():
        d, i = int(pid.split("_d")[1].split("_")[0]), int(pid.split("_i")[1].split("_")[0])
        expr = make_problem(d, i, task="arith").statement.split("Evaluate this Python expression. ", 1)[1]
        tree = ast.parse(expr, mode="eval").body
        meta[pid] = (n_ops(tree), height(tree))
    df["ops"] = df.problem_id.map(lambda p: meta[p][0])
    df["height"] = df.problem_id.map(lambda p: meta[p][1])
    print("ops check (generator depth == parsed op count):", (df.ops == df.depth).all())
    print("\nheight distribution per ops level (problems):")
    per = (
        df[df.arm == "B"]
        .drop_duplicates("problem_id")
        .groupby("depth")
        .height.describe()[["count", "min", "25%", "50%", "75%", "max"]]
    )
    print(per.to_string())
    print("\naccuracy by ops level x height (Astra; n in parentheses):")
    rows = []
    for me, dm in df.groupby("model_eff"):
        for (d, h), g in dm.groupby(["depth", "height"]):
            b = g[g.arm == "B"]
            f = g[(g.arm == "CB") & (g.k == 1000)]
            if len(b) < 5:
                continue
            rows.append(
                {
                    "model": me,
                    "ops": d,
                    "height": h,
                    "n": len(b),
                    "no_filler": b.correct.mean(),
                    "filler_1000": f.correct.mean() if len(f) else np.nan,
                }
            )
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(2).to_string(index=False))
    # partial association: within each ops level, Spearman of correctness with height
    from scipy.stats import spearmanr

    print("\nwithin-ops-level Spearman(correct, height):")
    for me, dm in df.groupby("model_eff"):
        for arm_k, g in dm.groupby(["arm", "k"]):
            parts = []
            for d, gg in g.groupby("depth"):
                if gg.height.nunique() > 1:
                    r, p = spearmanr(gg.correct.astype(int), gg.height)
                    parts.append(f"{d} ops: rho={r:+.2f} (p={p:.2g}, n={len(gg)})")
            print(f"  {me} {arm_k}: " + "; ".join(parts))
    res.to_csv(tag_dir("greenblatt") / "figs" / "arith_height.csv", index=False)
    # figure: Astra accuracy vs height, one line per ops level, no filler (dotted) vs filler 1000 (solid)
    a = res[res.model == "gpt-6-astra:low"]
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for d, g in a.groupby("ops"):
        c = ax.plot(g.height, g.filler_1000, marker="o", label=f"{d} ops, counting 1000")[0].get_color()
        ax.plot(g.height, g.no_filler, marker="x", ls=":", color=c, label=f"{d} ops, no filler")
    ax.set(
        xlabel="expression tree height (longest chain of nested operations)",
        ylabel="accuracy, no CoT",
        ylim=(-0.02, 1.02),
        title="gpt-6-astra:low: Gen-Arithmetic accuracy vs serial depth",
    )
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(tag_dir("greenblatt") / "figs" / "arith_height.png", dpi=150)
    print("wrote results/greenblatt/figs/arith_height.png / .csv")


if __name__ == "__main__":
    main()
