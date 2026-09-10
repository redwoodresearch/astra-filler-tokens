"""No-CoT serial-depth suite: accuracy vs depth per task, one line per model config, plus depth ceilings.
usage: uv run -m nf.suite suite"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import ROOT, load

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 12})
TASK_NAMES = {
    "perm": "permutation composition",
    "lookup": "iterated lookup table",
    "modchain": "mod-7 arithmetic chain",
    "reach": "d-hop reachability (yes/no)",
    "ca": "cellular automaton, cell 5",
    "stack": "stack machine, top",
    "xor": "parity of d flags",
}
SHORT = {
    "perm": "permutations",
    "lookup": "lookup table",
    "modchain": "mod-7 chain",
    "reach": "reachability",
    "ca": "cellular aut.",
    "stack": "stack machine",
    "xor": "parity",
}
COLORS = {
    "gpt-6-astra:low": "#D65F5F",
    "gpt-5.6-sol:none": "#4878CF",
    "gpt-5.6-sol:low": "#6ACC65",
    "gpt-5.5:none": "#B47CC7",
}


def ci95(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n)


def depth_ceiling(g: pd.DataFrame, thresh: float = 0.9) -> float:
    """Largest depth at which accuracy >= thresh, requiring all shallower depths to also pass (0 if none)."""
    ok = g.sort_values("depth")
    best = 0
    for d, acc in zip(ok.depth, ok.acc):
        if acc >= thresh:
            best = d
        else:
            break
    return best


def normalized_area(g: pd.DataFrame) -> float:
    """Mean over sampled depths of (acc - chance) / (1 - chance): a chance-corrected summary in [0, 1]."""
    return float(np.mean((g.acc - g.chance) / (1 - g.chance)))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["task", "model_eff", "depth"])
        .agg(
            n=("correct", "size"),
            acc=("correct", "mean"),
            chance=("chance", "first"),
            compliant=("compliant", "mean"),
            reason_tok=("reasoning_tokens", "mean"),
            incomplete=("status", lambda x: (x == "incomplete").mean()),
        )
        .reset_index()
    )
    return g


def figure(s: pd.DataFrame, out):
    tasks = [t for t in TASK_NAMES if t in set(s.task)]
    models = [m for m in COLORS if m in set(s.model_eff)] + sorted(set(s.model_eff) - set(COLORS))
    ncol = 4
    nrow = int(np.ceil(len(tasks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.2 * nrow), constrained_layout=True, squeeze=False)
    for ax, task in zip(axes.flat, tasks):
        st = s[s.task == task]
        for me in models:
            g = st[st.model_eff == me].sort_values("depth")
            if g.empty:
                continue
            ax.errorbar(
                g.depth, g.acc, yerr=ci95(g.acc, g.n), fmt="o-", lw=2, capsize=3, color=COLORS.get(me, "gray"), label=me
            )
        ax.axhline(st.chance.iloc[0], color="gray", ls=":", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(st.depth.unique()))
        ax.set_xticklabels([str(d) for d in sorted(st.depth.unique())])
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(TASK_NAMES[task], fontsize=12)
        ax.set_xlabel("depth d (log2 axis)", fontsize=11)
    for ax in axes.flat[len(tasks) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy, no CoT (↑)", fontsize=11)
    axes.flat[0].legend(frameon=False, fontsize=9, loc="lower left")
    n = int(s.n.median())
    fig.suptitle(
        f"No-CoT accuracy vs serial depth on seven synthetic tasks (n={n} per point, dotted = chance)", fontsize=14
    )
    fig.savefig(out, dpi=200)
    plt.close(fig)


def ceilings_figure(c: pd.DataFrame, out):
    tasks = [t for t in TASK_NAMES if t in set(c.task)]
    models = [m for m in COLORS if m in set(c.model_eff)]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    w = 0.8 / len(models)
    for j, me in enumerate(models):
        v = c[c.model_eff == me].set_index("task").reindex(tasks)
        x = np.arange(len(tasks)) + (j - (len(models) - 1) / 2) * w
        shown = v.norm_area.clip(lower=0)  # below-chance configs (truncated-reasoning artefacts) are drawn at 0
        ax.bar(x, shown, w, color=COLORS[me], label=me)
        for xi, val, raw in zip(x, shown, v.norm_area):
            ax.text(xi, val + 0.01, f"{raw:.2f}" if raw >= 0 else "<0", ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels([SHORT[t] for t in tasks], fontsize=11)
    ax.set_ylabel("chance-corrected accuracy averaged over sampled depths (↑)", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=10, ncol=len(models))
    ax.set_title("No-CoT serial-depth suite: per-task summary", fontsize=14)
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    tag = sys.argv[1]
    df = pd.concat([load(t) for t in sys.argv[1:]])
    df = df[df.status != "error"]
    out = ROOT / "results" / tag / "figs"
    out.mkdir(exist_ok=True)
    s = summarize(df)
    s.to_csv(out / "suite_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(f"{len(df)} records; compliance {df.compliant.mean():.3f}; max reasoning tokens {df.reasoning_tokens.max()}")
    print(s.pivot_table(index=["task", "model_eff"], columns="depth", values="acc").round(2).to_string())
    c = (
        s.groupby(["task", "model_eff"])
        .apply(
            lambda g: pd.Series(
                {"ceiling_90": depth_ceiling(g), "ceiling_75": depth_ceiling(g, 0.75), "norm_area": normalized_area(g)}
            ),
            include_groups=False,
        )
        .reset_index()
    )
    print("\nDepth ceiling (largest d with acc>=0.9 / >=0.75, monotone) and chance-corrected mean accuracy:")
    print(c.round(3).to_string(index=False))
    c.to_csv(out / "suite_ceilings.csv", index=False)
    figure(s, out / "suite_acc_vs_depth.png")
    ceilings_figure(c, out / "suite_summary.png")
    print("figs ->", out)
