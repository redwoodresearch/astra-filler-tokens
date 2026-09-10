"""Filler sensitivity by release date: how much each model gains, over the plain k=0 prompt, from user-supplied
problem copies (arm B) and from token-matched dot filler (arm DB) on the permutation task.
usage: uv run -m nf.filler_trend filler_trend"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import load, tag_dir
from .horizon import EXCLUDE, FAMILY_COLORS, attach_dates

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 12})


def gains(df: pd.DataFrame) -> pd.DataFrame:
    """Paired, chance-corrected gain over k=0 per model x arm x depth x k."""
    rows = []
    for me, d in df.groupby("model_eff"):
        base = d[d.k == 0].groupby("problem_id")["correct"].mean().astype(float)  # duplicates across tags averaged
        for (arm, depth, k), g in d[d.k > 0].groupby(["arm", "depth", "k"]):
            b = base.reindex(g.problem_id).values
            delta = (g.correct.astype(float).values - b) / (1 - g.chance.values)
            rows.append(
                {
                    "model_eff": me,
                    "arm": arm,
                    "depth": depth,
                    "k": k,
                    "n": len(g),
                    "acc": g.correct.mean(),
                    "k0_acc": np.nanmean(b),
                    "gain": np.nanmean(delta),
                    "se": np.nanstd(delta, ddof=1) / np.sqrt(len(g)),
                    "compliant": g.compliant.mean(),
                }
            )
    return pd.DataFrame(rows)


def summary(G: pd.DataFrame) -> pd.DataFrame:
    """Per model: mean gain over depths and k>0, per arm; plus slope of gain in log2 k."""
    rows = []
    for me, g in G.groupby("model_eff"):
        row = {"model_eff": me}
        for arm, name in (("B", "copies"), ("DB", "dots")):
            s = g[g.arm == arm]
            row[f"gain_{name}"] = s.gain.mean()
            row[f"gain_{name}_se"] = np.sqrt((s.se**2).sum()) / len(s)
            row[f"gain_{name}_k1"] = s[s.k == 1].gain.mean()
            row[f"gain_{name}_k8"] = s[s.k == 8].gain.mean()
            byk = s.groupby("k").gain.mean()
            row[f"slope_{name}"] = (
                np.polyfit(np.log2(byk.index.values.astype(float)), byk.values, 1)[0] if len(byk) > 1 else np.nan
            )
        row["k0_norm_acc"] = ((g.k0_acc - 0.125) / 0.875).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def fig_by_date(S: pd.DataFrame, out):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True, sharey=True)
    for ax, name, title in zip(
        axes,
        ("copies", "dots"),
        ("user-supplied problem copies (arm B)", "user-supplied dot filler, token-matched (arm DB)"),
    ):
        for fam, g in S.groupby("family"):
            ax.errorbar(
                g.release,
                g[f"gain_{name}"],
                yerr=1.96 * g[f"gain_{name}_se"],
                fmt="o",
                color=FAMILY_COLORS[fam],
                label=fam,
                capsize=2,
                ms=7,
                mec="white",
            )
        for _, r in S.iterrows():
            ax.annotate(r.label, (r.release, r[f"gain_{name}"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
        ax.axhline(0, color="gray", lw=1)
        ax.set_title(f"Gain from {title}", fontsize=13)
        ax.set_xlabel("release date", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(
        "chance-corrected accuracy gain over k=0\n(permutation task, mean over depths 3-8 and k=1,2,4,8)", fontsize=11
    )
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Filler sensitivity by release date (no CoT; paired against the same problems at k=0)", fontsize=14)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_dose(G: pd.DataFrame, S: pd.DataFrame, out, top: int = 12):
    """Gain vs k for the most filler-sensitive models (by copies gain), copies solid / dots dashed."""
    sel = S.sort_values("gain_copies", ascending=False).head(top)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True, sharey=True)
    cmap = plt.get_cmap("tab20")
    for i, (_, r) in enumerate(sel.iterrows()):
        for ax, arm in zip(axes, ("B", "DB")):
            g = G[(G.model_eff == r.model_eff) & (G.arm == arm)].groupby("k").gain.mean()
            ax.plot(g.index, g.values, "o-", color=cmap(i % 20), lw=2, label=r.label)
    for ax, t in zip(axes, ("user-supplied problem copies", "user-supplied dot filler")):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.axhline(0, color="gray", lw=1)
        ax.set_xlabel("k (filler length in problem-statement units)", fontsize=12)
        ax.set_title(f"Gain vs k: {t}", fontsize=13)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("chance-corrected gain over k=0 (mean over depths 3-8)", fontsize=11)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.concat([load(t) for t in sys.argv[1:]])
    df = df[(df.status != "error") & (df.task == "perm") & ~df.model_eff.isin(EXCLUDE) & df.depth.isin([3, 4, 6, 8])]
    out = tag_dir("horizon")
    out.mkdir(exist_ok=True)
    G = gains(df)
    S = attach_dates(summary(G)).dropna(subset=["release"]).sort_values("release")
    pd.set_option("display.width", 250)
    print(
        S[
            [
                "label",
                "release",
                "k0_norm_acc",
                "gain_copies",
                "gain_copies_k1",
                "gain_copies_k8",
                "slope_copies",
                "gain_dots",
                "gain_dots_k1",
                "gain_dots_k8",
                "slope_dots",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    G.to_csv(out / "filler_gains_by_cell.csv", index=False)
    S.to_csv(out / "filler_sensitivity_by_model.csv", index=False)
    fig_by_date(S, out / "filler_sensitivity_by_release_date.png")
    fig_dose(G, S, out / "filler_dose_response.png")
    print("figs ->", out)
