"""Filler dose-response across tasks: rows = model configs, columns = tasks; lines = depths (user-supplied solid,
model-emitted dashed). x = actual filler tokens.
usage: uv run -m nf.dots_tasks [--arms XB,XC|PB,PC] [--name dots|prefix] <tags...>"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import ROOT, load
from .dots_dose import se, to_dot_counts
from .suite import TASK_NAMES

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
MODELS = ["gpt-5.6-sol:none", "gpt-6-astra:low"]


def threshold(cell: pd.DataFrame, base: float) -> float:
    """Smallest dot count whose accuracy closes at least half the gap between the 0-dot baseline and 1.0."""
    target = base + 0.5 * (1 - base)
    hit = cell[(cell.k > 0) & (cell.acc >= target)].sort_values("k")
    return float(hit.k.iloc[0]) if len(hit) else np.inf


if __name__ == "__main__":
    args = sys.argv[1:]
    arms, name = ["XB", "XC"], "dots"
    if "--arms" in args:
        j = args.index("--arms")
        arms = args[j + 1].split(",")
        del args[j : j + 2]
    if "--name" in args:
        j = args.index("--name")
        name = args[j + 1]
        del args[j : j + 2]
    if "--models" in args:
        j = args.index("--models")
        MODELS = args[j + 1].split(",")
        del args[j : j + 2]
    U, M = (arms + [None])[:2]  # user-supplied, model-emitted arm names (M may be absent)
    df = pd.concat([load(t) for t in args])
    df = df[(df.status != "error") & df.arm.isin([x for x in ["B", U, M, "DB", "DC"] if x]) & df.model_eff.isin(MODELS)]
    # No-CoT validity filters: API refusals are a classifier artefact, not capability (Opus 5 at long fillers); calls with
    # any provider-reported reasoning tokens violate the no-CoT condition (Gemini 3.8 Flash at low effort). Both dropped.
    n0 = len(df)
    # Anthropic records carry an *estimated* hidden-token count (billed minus visible, ~4 by tokenizer offset), so use a
    # threshold there; OpenAI/OpenRouter report the field directly and 0 means none.
    rt = df.reasoning_tokens.fillna(0)
    ant = df.get("served_provider", pd.Series(index=df.index, dtype=object)).eq("anthropic")
    df = df[(df.status != "refusal") & ((rt <= 10) & ant | (rt == 0) & ~ant)]
    print(f"dropped {n0 - len(df)} of {n0} rows (refusals or reasoning_tokens>0)")
    if U != "B":
        df = df[(df.arm != "B") | (df.k == 0)]
    # Opus 5's model-emitted dot arm is invalid: 94% of those calls were refused by the API (reasoning_extraction).
    df = df[~((df.model_eff == "claude-opus-5:off") & (df.arm == "XC"))]
    if U == "B":  # copies mode: arm B at k>0 is k user-supplied copies; k=0 rows are the baseline
        df = df[df.task != "ca"]
        df.loc[(df.arm == "B") & (df.k > 0), "arm"] = "XB"
        U = "XB"
    if U == "XB" and name in ("dots", "plain"):
        df = to_dot_counts(df)
    if "filler_tokens" not in df:
        df["filler_tokens"] = df.k
    df["filler_tokens"] = df.filler_tokens.fillna(df.k)
    df["arm"] = df.arm.map({k: v for k, v in ((U, "XB"), (M, "XC")) if k}).fillna(df.arm)
    # Group by the target k (a full 40-problem cell); x = the cell's mean actual filler tokens.
    t = (
        df.groupby(["model_eff", "task", "depth", "arm", "k"])
        .agg(
            n=("correct", "size"),
            acc=("correct", "mean"),
            chance=("chance", "first"),
            compliant=("compliant", "mean"),
            x=("filler_tokens", "mean"),
        )
        .reset_index()
    )
    if name == "prefix":  # several targets can map to the same line-granular filler: merge them
        t["k"] = t.x.round().astype(int)
        t = t.groupby(["model_eff", "task", "depth", "arm", "k"], as_index=False).agg(
            n=("n", "sum"),
            acc=("acc", "mean"),
            chance=("chance", "first"),
            compliant=("compliant", "mean"),
            x=("x", "mean"),
        )
    tasks = [x for x in TASK_NAMES if x in set(t.task)]
    fig, axes = plt.subplots(
        len(MODELS), len(tasks), figsize=(4.3 * len(tasks), 4.6 * len(MODELS)), constrained_layout=True, squeeze=False
    )
    cmap = plt.get_cmap("viridis")
    rows = []
    for r, me in enumerate(MODELS):
        for c, task in enumerate(tasks):
            ax = axes[r, c]
            d = t[(t.model_eff == me) & (t.task == task)]
            depths = sorted(d.depth.unique())
            for i, depth in enumerate(depths):
                col = cmap(i / max(1, len(depths) - 1))
                base = d[(d.depth == depth) & (d.k == 0)]
                b = base.acc.iloc[0] if len(base) else np.nan
                ax.axhline(b, color=col, ls=":", lw=1)
                for arm, ls, mk, label in (("XB", "-", "o", "user"), ("XC", "--", "s", "model")):
                    sub = d[(d.depth == depth) & (d.arm == arm) & (d.k > 0)].sort_values("k")
                    if sub.empty:
                        continue
                    ax.errorbar(
                        sub.k,
                        sub.acc,
                        yerr=se(sub.acc, sub.n),
                        fmt=mk,
                        ls=ls,
                        color=col,
                        lw=1.8,
                        capsize=2,
                        ms=4,
                        label=f"d={depth} {label}",
                    )
                    rows.append(
                        {
                            "model_eff": me,
                            "task": task,
                            "depth": depth,
                            "arm": arm,
                            "k0_acc": b,
                            "threshold_tokens": threshold(sub, b),
                            "acc_at_max": sub.acc.iloc[-1],
                            "max_tokens": sub.k.iloc[-1],
                            "n_min": sub.n.min(),
                            "compliant": sub.compliant.mean(),
                        }
                    )
            if len(d):
                ax.axhline(d.chance.iloc[0], color="gray", lw=1, ls="-.")
                ax.set_xscale("log")
                ticks = [v for v in (1, 5, 20, 70, 140, 340, 700, 1400, 2800) if v <= d.k.max() * 1.05]
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(v) for v in ticks], fontsize=8)
                ax.legend(frameon=False, fontsize=7, ncol=2, loc="best")
            ax.set_ylim(-0.03, 1.05)
            if r == 0:
                ax.set_title(TASK_NAMES[task], fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{me}\naccuracy, no CoT (↑)", fontsize=10)
            if r == len(MODELS) - 1:
                ax.set_xlabel("number of copies, log" if name == "copies" else "filler tokens, log", fontsize=10)
    kind = {
        "dots": "Dot filler",
        "prefix": "Statement-prefix / repetition filler",
        "copies": "Whole-statement copies (user-supplied)",
    }.get(name, name)
    fig.suptitle(
        f"{kind} dose-response across tasks (dotted = same depth with no filler; dash-dot = chance; "
        "solid = user-supplied, dashed = model-emitted; n=40)",
        fontsize=12,
    )
    out = (
        ROOT
        / "results"
        / {"dots": "dots_tasks", "prefix": "prefix_tasks", "plain": "dots_plain"}.get(name, "horizon")
        / "figs"
    )
    out.mkdir(parents=True, exist_ok=True)
    suffix = (
        ""
        if MODELS == ["gpt-5.6-sol:none", "gpt-6-astra:low"]
        else "_" + "_".join(m.split("/")[-1].split(":")[0] for m in MODELS)
    )
    fig.savefig(out / f"{name}_across_tasks{suffix}.png", dpi=200)
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(R.round(2).to_string(index=False))
    R.to_csv(out / f"{name}_thresholds{suffix}.csv", index=False)
    print("fig ->", out / f"{name}_across_tasks{suffix}.png")
