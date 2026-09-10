"""Figures for a results/<tag> run. usage: uv run -m nf.plots main [depth_sweep_tag]"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import ROOT, load

matplotlib.use("Agg")
COL = {"B": "#4878CF", "C": "#D65F5F"}
MODEL_COL = {"gpt-6-astra:low": "#D65F5F", "gpt-5.6-sol:none": "#4878CF", "gpt-5.6-sol:low": "#6ACC65"}
LABEL = {"B": "user-supplied copies in prompt (B)", "C": "model-emitted copies before answer (C)"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 12})


def se(p, n):
    p, n = np.asarray(p, dtype=float), np.asarray(n, dtype=float)
    return 1.96 * np.sqrt(p * (1 - p) / n)


def acc_vs_k(df: pd.DataFrame, out: Path):
    """One figure per model config: accuracy vs k, panel per depth, arms B and C."""
    for me, d in df.groupby("model_eff"):
        depths = sorted(d.depth.unique())
        fig, axes = plt.subplots(1, len(depths), figsize=(3.4 * len(depths), 4.6), sharey=True, constrained_layout=True)
        for ax, depth in zip(np.atleast_1d(axes), depths):
            dd = d[d.depth == depth]
            for arm in ("B", "C"):
                sub = dd[(dd.arm == arm) | (dd.k == 0)].groupby("k")["correct"].agg(["mean", "size"]).reset_index()
                ax.errorbar(
                    sub.k,
                    sub["mean"],
                    yerr=se(sub["mean"], sub["size"]),
                    fmt="o-",
                    color=COL[arm],
                    capsize=3,
                    lw=2,
                    label=LABEL[arm],
                )
            ax.axhline(1 / 8, ls=":", color="gray", lw=1)
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xticks([0, 1, 2, 4, 8])
            ax.set_xticklabels(["0", "1", "2", "4", "8"])
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"depth {depth}", fontsize=13)
            ax.set_xlabel("k (verbatim copies of the problem)")
        np.atleast_1d(axes)[0].set_ylabel("Accuracy (↑)", fontsize=14)
        np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=10, frameon=False)
        n = int(d.groupby(["depth", "arm", "k"]).size().median())
        fig.suptitle(
            f"{me}: accuracy vs number of problem copies, by permutation depth (n={n} per point; dotted = chance)",
            fontsize=14,
        )
        fig.savefig(out / f"acc_vs_k__{me.replace(':', '_')}.png", dpi=200)
        plt.close(fig)


def gap_vs_k(df: pd.DataFrame, out: Path, min_depth: int = 4):
    """Paired C-B accuracy delta vs k: one panel per model config, one line per depth."""
    models = sorted(df.model_eff.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 5.2), sharey=True, constrained_layout=True)
    depths = sorted(d for d in df.depth.unique() if d >= min_depth)
    cmap = plt.get_cmap("viridis")
    rows = []
    for ax, me in zip(np.atleast_1d(axes), models):
        d = df[df.model_eff == me]
        for i, depth in enumerate(depths):
            dd = d[(d.depth == depth) & (d.k > 0)]
            piv = dd.pivot_table(index=["problem_id", "k"], columns="arm", values="correct", aggfunc="first").dropna()
            delta = (piv["C"].astype(float) - piv["B"].astype(float)).groupby(level="k")
            m, s, n = delta.mean(), delta.std() / np.sqrt(delta.size()), delta.size()
            ax.errorbar(
                m.index * (1 + 0.04 * (i - len(depths) / 2)),
                m.values,
                yerr=1.96 * s.values,
                fmt="o-",
                lw=2,
                capsize=3,
                color=cmap(i / max(1, len(depths) - 1)),
                label=f"depth {depth}",
            )
            for k in m.index:
                rows.append({"model_eff": me, "depth": depth, "k": k, "delta": m[k], "se": s[k], "n": n[k]})
        ax.axhline(0, color="gray", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_title(me, fontsize=14)
        ax.set_xlabel("k (verbatim copies of the problem)", fontsize=12)
    np.atleast_1d(axes)[0].set_ylabel(
        "Accuracy: model-emitted (C) minus user-supplied (B) copies\n(paired, 95% CI)", fontsize=12
    )
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=11)
    n = int(pd.DataFrame(rows).n.median())
    fig.suptitle(f"C minus B accuracy gap vs k, by permutation depth (n={n} paired problems per point)", fontsize=14)
    fig.savefig(out / "gap_vs_k.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "gap_vs_k.csv", index=False)
    return pd.DataFrame(rows)


def reasoning_split(df: pd.DataFrame, out: Path, me: str = "gpt-5.6-sol:low"):
    """For one config: C-arm accuracy split by whether hidden reasoning tokens were emitted, vs B-arm accuracy."""
    d = df[df.model_eff == me]
    c = d[d.arm == "C"].copy()
    c["rt"] = np.where(c.reasoning_tokens > 0, "C, hidden reasoning > 0", "C, hidden reasoning = 0")
    b = d[(d.arm == "B") & (d.k > 0)].copy()
    b["rt"] = "B (user-supplied copies)"
    g = pd.concat([c, b]).groupby(["depth", "rt"])["correct"].agg(["mean", "size"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    cats = ["B (user-supplied copies)", "C, hidden reasoning = 0", "C, hidden reasoning > 0"]
    cols = ["#4878CF", "#D65F5F", "#B47CC7"]
    depths = sorted(g.depth.unique())
    w = 0.26
    for j, (cat, col) in enumerate(zip(cats, cols)):
        sub = g[g.rt == cat].set_index("depth").reindex(depths)
        x = np.arange(len(depths)) + (j - 1) * w
        ax.bar(
            x, sub["mean"], w, yerr=se(sub["mean"].fillna(0), sub["size"].fillna(1)), capsize=3, color=col, label=cat
        )
        for xi, (m, n) in zip(x, zip(sub["mean"], sub["size"])):
            if pd.notna(m):
                ax.text(xi, m + 0.03, f"{m:.2f}\nn={int(n)}", ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(depths)))
    ax.set_xticklabels([f"depth {d}" for d in depths])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy (↑), k>0 pooled", fontsize=14)
    ax.set_title(f"{me}: accuracy by arm and by presence of hidden reasoning tokens", fontsize=14)
    ax.legend(frameon=False, fontsize=11, loc="upper left", ncol=3)
    fig.savefig(out / f"reasoning_split__{me.replace(':', '_')}.png", dpi=200)
    plt.close(fig)


def depth_ceiling(dfs: list[pd.DataFrame], out: Path):
    """k=0 accuracy vs depth per model, pooling all k=0 records passed in."""
    d = pd.concat(dfs)
    d = d[d.k == 0]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for me, g in d.groupby("model_eff"):
        s = g.groupby("depth")["correct"].agg(["mean", "size"]).reset_index()
        ax.errorbar(
            s.depth,
            s["mean"],
            yerr=se(s["mean"], s["size"]),
            fmt="o-",
            lw=2,
            capsize=3,
            color=MODEL_COL.get(me),
            label=me,
        )
    ax.axhline(1 / 8, ls=":", color="gray", lw=1)
    ax.set_xlabel("Permutation depth d (number of composed permutations)", fontsize=14)
    ax.set_ylabel("Accuracy with no chain of thought, k=0 (↑)", fontsize=14)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("No-CoT accuracy vs serial depth: gpt-6-astra vs gpt-5.6-sol (dotted = chance)", fontsize=15)
    ax.legend(frameon=False, fontsize=12)
    fig.savefig(out / "depth_ceiling.png", dpi=200)
    plt.close(fig)


def gain_vs_baseline(df: pd.DataFrame, out: Path, min_depth: int = 3):
    """Paired accuracy gain of C_k (model-emitted copies) and B_k (user-supplied copies) over the plain
    no-reasoning baseline (k=0: statement once, answer immediately). One panel per model config."""
    models = sorted(df.model_eff.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 5.2), sharey=True, constrained_layout=True)
    depths = sorted(d for d in df.depth.unique() if d >= min_depth)
    cmap = plt.get_cmap("viridis")
    rows = []
    for ax, me in zip(np.atleast_1d(axes), models):
        d = df[df.model_eff == me]
        base = d[d.k == 0].set_index("problem_id")["correct"].astype(float)
        for i, depth in enumerate(depths):
            for arm, ls in (("C", "-"), ("B", "--")):
                dd = d[(d.depth == depth) & (d.k > 0) & (d.arm == arm)].copy()
                dd["delta"] = dd["correct"].astype(float).values - base.reindex(dd.problem_id).values
                g = dd.groupby("k")["delta"]
                m, s, n = g.mean(), g.std() / np.sqrt(g.size()), g.size()
                ax.errorbar(
                    m.index * (1 + 0.04 * (i - len(depths) / 2)),
                    m.values,
                    yerr=1.96 * s.values,
                    fmt="o" + ls,
                    lw=2,
                    capsize=3,
                    color=cmap(i / max(1, len(depths) - 1)),
                    label=f"depth {depth}, {'model-emitted (C)' if arm == 'C' else 'user-supplied (B)'}",
                )
                for k in m.index:
                    rows.append(
                        {"model_eff": me, "depth": depth, "arm": arm, "k": k, "gain": m[k], "se": s[k], "n": n[k]}
                    )
        ax.axhline(0, color="gray", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_title(me, fontsize=14)
        ax.set_xlabel("k (verbatim copies of the problem)", fontsize=12)
    np.atleast_1d(axes)[0].set_ylabel("Accuracy gain over plain no-reasoning eval (k=0)\n(paired, 95% CI)", fontsize=12)
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=8, loc="best", ncol=1)
    fig.suptitle(
        "Gain over the k=0 baseline: solid = model-emitted copies (C), dashed = user-supplied copies (B)", fontsize=14
    )
    fig.savefig(out / "gain_vs_baseline.png", dpi=200)
    plt.close(fig)
    r = pd.DataFrame(rows)
    r.to_csv(out / "gain_vs_baseline.csv", index=False)
    return r


if __name__ == "__main__":
    tag = sys.argv[1]
    out = ROOT / "results" / tag / "figs"
    out.mkdir(exist_ok=True)
    df = load(tag)
    df = df[df.status != "error"]
    acc_vs_k(df, out)
    gap_vs_k(df, out)
    reasoning_split(df, out)
    gain_vs_baseline(df, out)
    extra = [load(t) for t in sys.argv[2:]]
    depth_ceiling([df, *extra], out)
    print("figs ->", out)
