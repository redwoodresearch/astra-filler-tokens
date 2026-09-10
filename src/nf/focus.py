"""Focused depth curves: Astra vs Fable 5.1 vs Opus 5 vs Sol on the six clean tasks.
usage: uv run -m nf.focus suite sweep_ant"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from .analyze import load, tag_dir
from .suite import TASK_NAMES, ci95

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 12})
SEL = {
    "gpt-6-astra:low": ("gpt-6-astra (low, prompted-empty CoT)", "#D65F5F"),
    "claude-fable-5-1:low": ("claude-fable-5-1 (low, thinking always-on but unused)", "#4878CF"),
    "claude-opus-5:off": ("claude-opus-5 (thinking disabled)", "#6ACC65"),
    "gpt-5.6-sol:none": ("gpt-5.6-sol (effort none)", "#B47CC7"),
}

if __name__ == "__main__":
    df = pd.concat([load(t) for t in sys.argv[1:]])
    df = df[(df.status != "error") & (df.task != "ca") & df.model_eff.isin(SEL) & (df.depth <= 16)]
    s = (
        df.groupby(["task", "model_eff", "depth"])
        .agg(n=("correct", "size"), acc=("correct", "mean"), chance=("chance", "first"))
        .reset_index()
    )
    tasks = [t for t in TASK_NAMES if t in set(s.task)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for ax, task in zip(axes.flat, tasks):
        st = s[s.task == task]
        for me, (label, col) in SEL.items():
            g = st[st.model_eff == me].sort_values("depth")
            ax.errorbar(g.depth, g.acc, yerr=ci95(g.acc, g.n), fmt="o-", lw=2, capsize=3, color=col, label=label)
        ax.axhline(st.chance.iloc[0], color="gray", ls=":", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(st.depth.unique()))
        ax.set_xticklabels([str(d) for d in sorted(st.depth.unique())])
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(TASK_NAMES[task], fontsize=12)
        ax.set_xlabel("depth d", fontsize=11)
    axes[0, 0].set_ylabel("accuracy, no CoT (↑)", fontsize=11)
    axes[1, 0].set_ylabel("accuracy, no CoT (↑)", fontsize=11)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(
        "No-CoT accuracy vs depth: gpt-6-astra vs claude-fable-5-1 vs claude-opus-5 vs gpt-5.6-sol (n=40 per point)",
        fontsize=13,
    )
    out = tag_dir("horizon") / "focus_astra_vs_fable.png"
    fig.savefig(out, dpi=200)
    print("fig ->", out)
    print(s.pivot_table(index=["task", "model_eff"], columns="depth", values="acc").round(2).to_string())
