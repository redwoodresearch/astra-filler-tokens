"""Greenblatt (2025) replication analysis: no-CoT accuracy vs filler on AIME/HMMT and Gen-Arithmetic.
usage: uv run -m nf.greenblatt greenblatt"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .analyze import load, tag_dir
from .tasks import _dataset

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
ARMS = {
    "CB": ("counting filler 1..N", "#6ACC65", "o", "-"),
    "XB": ("300 dots", "#4878CF", "D", "none"),
    "RB": ("question repeats", "#D65F5F", "s", "--"),
}
# rows of the figure: (task, depth filter or None) -> label
ROWS = {
    ("aime", None): "AIME/HMMT 2024-26",
    ("aimepp", "AIME"): "AIME-Plus-Plus (Aug 2026), AIME tier",
    ("aimepp", None): "AIME-Plus-Plus (Aug 2026), all tiers",
    ("arith", (5, 6, 7)): "Gen-Arithmetic, 5-7 ops",
    ("arith", (10,)): "Gen-Arithmetic, 10 ops",
    ("arith", (15,)): "Gen-Arithmetic, 15 ops",
}
TASKS = {t for t, _ in ROWS} | {"lehigh26"}
MODEL_LABELS = {
    "gpt-6-astra:low": "gpt-6-astra:low (prompted no-CoT)",
    "claude-opus-4-5-20251101:off": "claude-opus-4.5 (thinking off)",
    "claude-opus-5:off": "claude-opus-5 (thinking off)",
    "gpt-5.6-sol:none": "gpt-5.6-sol:none",
    "deepseek/deepseek-v3.2:off": "deepseek-v3.2 (reasoning off)",
}
TIER = [r["tier"] for r in _dataset("aimepp")]

if __name__ == "__main__":
    args = sys.argv[1:]
    compact = "--compact" in args  # 4x4: drop the all-tiers and 5-7-op rows
    args = [a for a in args if a != "--compact"]
    if compact:
        ROWS = {k: v for k, v in ROWS.items() if k not in {("arith", (10,)), ("arith", (5, 6, 7))}}
    df = pd.concat([load(t) for t in args]).reset_index(drop=True)
    df = df[(df.status != "error") & df.task.isin(TASKS) & df.arm.isin(["B", "CB", "XB", "RB"])]
    df = df[(df.arm != "B") | (df.k == 0)]
    rt = df.reasoning_tokens.fillna(0)
    ant = df.get("served_provider", pd.Series(index=df.index, dtype=object)).eq("anthropic")
    df = df[(df.status != "refusal") & ((rt <= 10) & ant | (rt == 0) & ~ant)]
    df["x"] = np.where(df.arm == "B", 0, df.get("filler_tokens", df.k).fillna(df.k))
    done = (
        df[df.arm == "B"].groupby("model_eff").size()
    )  # only models whose runs completed (Sol / Opus 5 were stopped early)
    models = [
        m
        for m in [
            "gpt-6-astra:low",
            "claude-opus-4-5-20251101:off",
            "claude-opus-5:off",
            "gpt-5.6-sol:none",
            "deepseek/deepseek-v3.2:off",
        ]
        if done.get(m, 0) >= 200
    ]
    rows = []
    trend_rows = []
    fig, axes = plt.subplots(
        len(ROWS), len(models), figsize=(5 * len(models), 4.2 * len(ROWS)), constrained_layout=True, squeeze=False
    )
    for r, ((task, depths), tname) in enumerate(ROWS.items()):
        for c, me in enumerate(models):
            ax = axes[r, c]
            d = df[(df.model_eff == me) & (df.task == task)]
            if isinstance(depths, str):  # tier filter for aimepp
                d = d[d.problem_id.map(lambda pid: TIER[int(pid.split("_i")[1].split("_")[0])]) == depths]
            elif depths is not None:
                d = d[d.depth.isin(depths)]
            if d[d.arm == "B"].empty:
                ax.set_visible(False)
                continue
            base = d[d.arm == "B"].set_index("problem_id").correct.astype(float)
            b, nb = base.mean(), len(base)
            ax.axhline(b, color="gray", ls=":", lw=1.2, label=f"no filler ({b:.2f})")
            for arm, (lab, col, mk, ls) in ARMS.items():
                s = d[d.arm == arm]
                if s.empty:
                    continue
                g = (
                    s.groupby("k")
                    .agg(x=("x", "mean"), acc=("correct", "mean"), n=("correct", "size"))
                    .reset_index()
                    .sort_values("x")
                )
                g = g[g.n >= 0.5 * nb]  # drop cells the model mostly refused (Opus 5 on some doses)
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
                for _, row in g.iterrows():
                    cell = s[s.k == row.k]
                    delta = cell.correct.astype(float).values - base.reindex(cell.problem_id).values
                    delta = delta[~np.isnan(delta)]
                    rows.append(
                        {
                            "model_eff": me,
                            "task": task,
                            "depths": str(depths),
                            "arm": arm,
                            "k": row.k,
                            "filler_tokens": round(row.x),
                            "n": len(delta),
                            "acc_k0": b,
                            "acc": row.acc,
                            "paired_gain": delta.mean(),
                            "gain_se": delta.std(ddof=1) / np.sqrt(len(delta)) if len(delta) > 1 else np.nan,
                            "p_paired": 2
                            * (
                                1
                                - __import__("scipy.stats").stats.norm.cdf(
                                    abs(delta.mean()) / (delta.std(ddof=1) / np.sqrt(len(delta)))
                                )
                            )
                            if len(delta) > 1 and delta.std() > 0
                            else np.nan,
                        }
                    )
                if arm == "CB":
                    cell = pd.concat([d[d.arm == "B"], s])
                    tau, p = kendalltau(cell.k.values, cell.correct.astype(int).values)
                    trend_rows.append({"model_eff": me, "task": task, "depths": str(depths), "tau": tau, "p": p})
            ax.set_xscale("symlog", linthresh=10)
            ax.set_xlim(-1, 4000)
            ax.set_ylim(-0.02, 1.02)
            if r == 0 or not any(axes[rr, c].get_visible() for rr in range(r)):
                ax.set_title(MODEL_LABELS.get(me, me), fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{tname}\naccuracy, no CoT", fontsize=10)
            if r == len(ROWS) - 1:
                ax.set_xlabel("filler tokens (symlog)", fontsize=10)
            top = max(
                [b]
                + [
                    float(np.nanmax(np.asarray(ln.get_ydata(), dtype=float)))
                    for ln in ax.get_lines()
                    if len(ln.get_ydata())
                ]
            )
            ax.legend(frameon=False, fontsize=8, loc="lower right" if top > 0.55 else "upper right")
    fig.suptitle(
        "Prompted no-CoT accuracy vs user-supplied filler",
        fontsize=12,
    )
    out = tag_dir("greenblatt") / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / ("greenblatt_replication_4x4.png" if compact else "greenblatt_replication.png"), dpi=170)
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(R.round(3).to_string(index=False))
    R.to_csv(out / "greenblatt_replication.csv", index=False)
    pd.DataFrame(trend_rows).to_csv(out / "greenblatt_counting_trend.csv", index=False)
    print("fig ->", out / ("greenblatt_replication_4x4.png" if compact else "greenblatt_replication.png"))
