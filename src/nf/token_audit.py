"""Audit hidden reasoning in the dose runs: provider-reported reasoning tokens and total output tokens vs filler length.
usage: uv run -m nf.token_audit --name dots dots_dose dots_tasks fillerD_astra fillerD_sol
       uv run -m nf.token_audit --name prefix --arms PB,PC prefix_tasks"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import load, tag_dir
from .dots_dose import to_dot_counts

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
MODELS = ["gpt-5.6-sol:none", "gpt-6-astra:low"]
FLOOR = 0.3  # zeros are drawn at this value on the log axis

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
    U, M = arms
    df = pd.concat([load(t) for t in args])
    df = df[(df.status != "error") & df.model_eff.isin(MODELS) & df.arm.isin(["B", U, M, "DB", "DC"])]
    df = df[(df.arm != "B") | (df.k == 0)]
    if name == "dots":
        df = to_dot_counts(df)
    if "filler_tokens" not in df:
        df["filler_tokens"] = df.k
    df["x"] = np.where(df.arm == "B", 0, df.filler_tokens.fillna(df.k))
    df["arm"] = df.arm.map({U: "XB", M: "XC"}).fillna(df.arm)
    df["reasoning_tokens"] = df.reasoning_tokens.fillna(0)
    g = (
        df.groupby(["model_eff", "arm", "x"])
        .agg(
            n=("correct", "size"),
            reason_mean=("reasoning_tokens", "mean"),
            reason_max=("reasoning_tokens", "max"),
            reason_nonzero=("reasoning_tokens", lambda v: (v > 0).mean()),
            out_mean=("output_tokens", "mean"),
            out_max=("output_tokens", "max"),
            expected=("expected_output_tokens", "mean"),
            incomplete=("status", lambda s: (s != "completed").mean()),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(len(MODELS), 2, figsize=(14, 4.8 * len(MODELS)), constrained_layout=True, squeeze=False)
    for r, me in enumerate(MODELS):
        d = g[g.model_eff == me]
        ax = axes[r, 0]
        for arm, col, lab in (
            ("B", "gray", "no filler (k=0)"),
            ("XB", "#4878CF", "user-supplied filler"),
            ("XC", "#D65F5F", "model-emitted filler"),
        ):
            s = d[d.arm == arm].sort_values("x")
            if s.empty:
                continue
            xs = s.x.clip(lower=0.7)
            ax.plot(xs, s.out_mean.clip(lower=FLOOR), "o-", color=col, label=f"{lab}: mean output tokens")
            ax.plot(xs, s.out_max.clip(lower=FLOOR), "^", color=col, alpha=0.5, ms=5, label=f"{lab}: max")
        s = d[d.arm == "XC"].sort_values("x")
        if len(s):
            ax.plot(
                s.x.clip(lower=0.7),
                s.expected.clip(lower=FLOOR),
                "k--",
                lw=1,
                label="expected compliant output (filler + answer)",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{me}: total output tokens (billed) vs filler length", fontsize=11)
        ax.set_xlabel("filler length (tokens), log; k=0 drawn at 0.7")
        ax.set_ylabel("output tokens, log")
        ax.legend(frameon=False, fontsize=8)
        ax = axes[r, 1]
        for arm, col, lab in (
            ("B", "gray", "no filler"),
            ("XB", "#4878CF", "user-supplied"),
            ("XC", "#D65F5F", "model-emitted"),
        ):
            s = d[d.arm == arm].sort_values("x")
            if s.empty:
                continue
            xs = s.x.clip(lower=0.7)
            ax.plot(xs, s.reason_mean.clip(lower=FLOOR), "o-", color=col, label=f"{lab}: mean reasoning tokens")
            ax.plot(xs, s.reason_max.clip(lower=FLOOR), "^", color=col, alpha=0.5, ms=5, label=f"{lab}: max")
        ax.axhline(FLOOR, color="k", lw=0.8, ls=":")
        ax.text(0.75, FLOOR * 1.15, "= 0 (drawn at floor)", fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(FLOOR * 0.7, max(10, d.reason_max.max() * 2 if len(d) else 10))
        ax.set_title(f"{me}: provider-reported reasoning tokens vs filler length", fontsize=11)
        ax.set_xlabel("filler length (tokens), log; k=0 drawn at 0.7")
        ax.set_ylabel("reasoning tokens (usage field), log")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Hidden-reasoning audit for the {name} dose runs (every call; n per point = 40 x depths x tasks)", fontsize=12
    )
    out = tag_dir("horizon") / "figs" / f"token_audit_{name}.png"
    fig.savefig(out, dpi=200)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(g.round(2).to_string(index=False))
    tot = df.groupby("model_eff").agg(
        calls=("correct", "size"),
        reason_sum=("reasoning_tokens", "sum"),
        reason_max=("reasoning_tokens", "max"),
        calls_with_reasoning=("reasoning_tokens", lambda v: int((v > 0).sum())),
        incomplete=("status", lambda s: int((s != "completed").sum())),
    )
    print("\n", tot.to_string())
    print("fig ->", out)
