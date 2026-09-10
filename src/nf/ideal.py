"""The 'ideal graphs': accuracy vs filler tokens on a log axis, using the exact-dot cells at powers of four (plus 8,192)
and the no-filler baseline (plotted at x=1).

  1. nhop_astra.png       Astra, n-hop: one line per hop count.
  2. nhop_models_4hop.png five models at 4 hops: accuracy vs filler tokens.
  3. nhop_models_hops.png five models at ~2,000 filler tokens (counting 1..1000) and none: accuracy vs hops.
  4. arith15_models.png   Gen-Arithmetic 15 ops, five models: accuracy vs filler tokens.
  5. aimepp_models.png    AIME-Plus-Plus AIME tier, five models.   6. aime_models.png  public AIME/HMMT, five models.
Usage: uv run -m nf.ideal"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import ROOT, aimepp_tier, load, tag_dir

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
MODELS = {
    "gpt-6-astra:low": ("gpt-6-astra", "#D62728"),
    "gpt-5.6-sol:none": ("gpt-5.6-sol", "#1F77B4"),
    "claude-opus-5:off": ("claude-opus-5", "#2CA02C"),
    "claude-opus-4-5-20251101:off": ("claude-opus-4.5", "#9467BD"),
    "deepseek/deepseek-v3.2:off": ("deepseek-v3.2", "#8C564B"),
}
OUT = tag_dir("ideal")


def _valid(df):
    df = df[df.status.isin(["completed"]) & df.model_eff.isin(MODELS)]
    ant = df.get("served_provider", pd.Series(index=df.index, dtype=object)).eq("anthropic")
    rt = df.reasoning_tokens.fillna(0)
    df = df[((rt <= 10) & ant) | ((rt == 0) & ~ant)]
    doses = {
        4,
        16,
        64,
        256,
        1024,
        4096,
        8192,
        16384,
    }  # exact-dot cells only, one point per dose (counting cells stay in the tables)
    df = df[((df.arm == "B") & (df.k == 0)) | ((df.arm == "XB") & df.k.isin(doses)) | (df.arm == "CB")].copy()
    df["x"] = np.where(df.arm == "B", 0, df.filler_tokens.fillna(df.k))
    df["correct"] = df.correct.astype(float)
    return df


def _curve(ax, d, label, color, marker="o", zorder=2):
    """One accuracy-vs-tokens line: baseline at x=1, then one point per dose. Every exact-dot cell is kept; a counting
    cell is kept only when no dot cell lies within a factor of two of its token count. Cells the model mostly refused
    (Opus 5 on some doses) or that were only partly run are dropped first."""
    cells = d.groupby(["arm", "x"]).correct.agg(["mean", "size"]).reset_index()
    if cells.empty:
        return
    cells = cells[cells["size"] >= 0.5 * cells["size"].max()]
    xb = cells[cells.arm == "XB"].x.to_numpy()
    keep = cells.apply(lambda r: r.arm != "CB" or not any(abs(np.log2(r.x / v)) < 1 for v in xb if v > 0), axis=1)
    g = cells[keep].groupby("x")[["mean", "size"]].first().sort_index().astype(float)
    ax.errorbar(
        g.index,
        g["mean"],
        yerr=1.96 * np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"]),
        marker=marker,
        ms=4,
        capsize=2,
        color=color,
        label=label,
        zorder=zorder,
    )


def _style(ax, title):
    ax.set_xscale("symlog", linthresh=4)
    ax.set(xlabel="filler tokens", ylabel="accuracy, no CoT", ylim=(-0.02, 1.02), title=title)
    ax.grid(alpha=0.3)


def main(nhop_task: str = "nhop"):
    """nhop_task: "nhop" (original parenthesised phrasing) or "nhopnl" (rewritten nested English)."""
    global OUT
    OUT = ROOT / "results" / "main" / ("ideal" if nhop_task == "nhop" else "ideal_nl")
    OUT.mkdir(parents=True, exist_ok=True)
    nh = _valid(load("nhop"))
    nh = nh[nh.task == nhop_task]  # both phrasings share the tag
    gb = _valid(pd.concat([load("greenblatt"), load("greenblatt2")]).reset_index(drop=True))
    # 1. Astra n-hop, line per hop count
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    a = nh[nh.model_eff == "gpt-6-astra:low"]
    cmap = plt.get_cmap("viridis")
    for h, d in a.groupby("depth"):
        _curve(ax, d, f"{h} hops", cmap((h - 2) / 5))
    _style(ax, "gpt-6-astra (prompted no-CoT): N-hop natural facts")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(OUT / "nhop_astra.png", dpi=150)
    # 2. five models at 4 hops
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for me, (lab, col) in MODELS.items():
        _curve(ax, nh[(nh.model_eff == me) & (nh.depth == 4)], lab, col, zorder=10 if "astra" in me else 2)
    _style(ax, "N-hop natural facts, 4 hops: accuracy vs filler tokens")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(OUT / "nhop_models_4hop.png", dpi=150)
    # 3. five models, accuracy vs hops at no filler (dotted) and counting 1..1000 (solid)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for me, (lab, col) in MODELS.items():
        d = nh[nh.model_eff == me]
        for arm, k, ls, mk, suffix in [
            ("B", 0, ":", "x", ", no filler"),
            ("CB", 1000, "-", "o", ", ~2,000 filler tokens"),
        ]:
            g = d[(d.arm == arm) & (d.k == k)].groupby("depth").correct.agg(["mean", "size"])
            if g.empty:
                continue
            ax.errorbar(
                g.index,
                g["mean"],
                yerr=1.96 * np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"]),
                ls=ls,
                marker=mk,
                ms=4,
                capsize=2,
                color=col,
                label=lab if arm == "CB" else None,
            )
    ax.set(
        xlabel="hops",
        ylabel="accuracy, no CoT",
        ylim=(-0.02, 1.02),
        title="N-hop natural facts: accuracy vs hops (dotted: no filler; solid: ~2,000 filler tokens)",
    )
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(OUT / "nhop_models_hops.png", dpi=150)
    # 4-6. math, five models
    for name, d0, title in [
        (
            "arith15_models",
            gb[(gb.task == "arith") & (gb.depth == 15)],
            "Gen-Arithmetic, 15 ops: accuracy vs filler tokens",
        ),
        (
            "aimepp_models",
            gb[gb.task == "aimepp"][lambda d: d.problem_id.map(aimepp_tier) == "AIME"],
            "AIME-Plus-Plus, AIME tier: accuracy vs filler tokens",
        ),
        ("aime_models", gb[gb.task == "aime"], "AIME/HMMT 2024-26: accuracy vs filler tokens"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        for me, (lab, col) in MODELS.items():
            _curve(ax, d0[d0.model_eff == me], lab, col, zorder=10 if "astra" in me else 2)
        _style(ax, title)
        ax.legend(frameon=False, fontsize=9, loc="upper left" if name == "arith15_models" else "best")
        fig.savefig(OUT / f"{name}.png", dpi=150)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "nhop")
