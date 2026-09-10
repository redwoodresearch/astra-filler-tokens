"""Four-arm comparison figure: gain over the plain no-reasoning eval (k=0) for
  B  = user-supplied problem copies      C  = model-emitted problem copies
  DB = user-supplied dot filler          DC = model-emitted dot filler (token-matched)
Rows = model configs, columns = depths. usage: uv run -m nf.plot_arms main fillerD_astra fillerD_sol"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import ROOT, load

matplotlib.use("Agg")
ARMS = {
    "B": ("user-supplied problem copies", "#4878CF", "--"),
    "C": ("model-emitted problem copies", "#D65F5F", "-"),
    "DB": ("user-supplied dot filler", "#4878CF", "-."),
    "DC": ("model-emitted dot filler", "#D65F5F", ":"),
}
PRETTY = {
    "gpt-6-astra:low": "gpt-6-astra (effort=low, prompted-empty CoT)",
    "gpt-5.6-sol:none": "gpt-5.6-sol (effort=none)",
}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 12})


def gains(df: pd.DataFrame) -> pd.DataFrame:
    """Per model/depth/arm/k: paired gain over that model's k=0 accuracy on the same problems."""
    rows = []
    for me, d in df.groupby("model_eff"):
        base = d[d.k == 0].set_index("problem_id")["correct"].astype(float)
        for (depth, arm, k), g in d[d.k > 0].groupby(["depth", "arm", "k"]):
            delta = g["correct"].astype(float).values - base.reindex(g.problem_id).values
            delta = delta[~np.isnan(delta)]
            rows.append(
                {
                    "model_eff": me,
                    "depth": depth,
                    "arm": arm,
                    "k": k,
                    "n": len(delta),
                    "acc": g.correct.mean(),
                    "k0_acc": base.reindex(g.problem_id).mean(),
                    "gain": delta.mean(),
                    "se": delta.std(ddof=1) / np.sqrt(len(delta)) if len(delta) > 1 else np.nan,
                    "compliant": g.compliant.mean(),
                    "reason_tok": g.reasoning_tokens.mean(),
                }
            )
    return pd.DataFrame(rows)


def figure(df: pd.DataFrame, out, models=("gpt-5.6-sol:none", "gpt-6-astra:low")):
    G = gains(df)
    per_model = {me: sorted(G[(G.model_eff == me) & G.arm.isin(["DB", "DC"])].depth.unique()) for me in models}
    ncol = max(1, max(len(v) for v in per_model.values()))
    fig, axes = plt.subplots(
        len(models), ncol, figsize=(4.2 * ncol, 4.6 * len(models)), constrained_layout=True, squeeze=False, sharey=True
    )
    for r, me in enumerate(models):
        for c in range(ncol):
            ax = axes[r, c]
            if c >= len(per_model[me]):
                ax.set_visible(False)
                continue
            depth = per_model[me][c]
            for arm, (label, col, ls) in ARMS.items():
                s = G[(G.model_eff == me) & (G.depth == depth) & (G.arm == arm)].sort_values("k")
                if s.empty:
                    continue
                ax.errorbar(s.k, s.gain, yerr=1.96 * s.se, fmt="o", ls=ls, color=col, lw=2, capsize=3, label=label)
            k0 = G[(G.model_eff == me) & (G.depth == depth)].k0_acc.iloc[0]
            ax.axhline(0, color="gray", lw=1)
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8])
            ax.set_xticklabels(["1", "2", "4", "8"])
            ax.set_ylim(-0.35, 1.05)
            ax.set_title(f"depth {depth}  (k=0 acc {k0:.2f})", fontsize=12)
            ax.set_xlabel("k (filler length, in problem-statement units)", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{PRETTY.get(me, me)}\n\ngain over k=0 (paired, 95% CI)", fontsize=11)
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(
        "Accuracy gain over plain no-reasoning eval: problem copies vs token-matched dot filler, "
        "user-supplied vs model-emitted",
        fontsize=14,
    )
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return G


if __name__ == "__main__":
    df = pd.concat([load(t) for t in sys.argv[1:]])
    df = df[df.status != "error"]
    out = ROOT / "results" / sys.argv[1] / "figs" / "arms_BCD_vs_baseline.png"
    G = figure(df, out)
    pd.set_option("display.width", 220)
    G = G[G.arm.isin(["DB", "DC"])].sort_values(["model_eff", "depth", "arm", "k"])
    print(G.round(3).to_string(index=False))
    G.to_csv(out.with_suffix(".csv"), index=False)
    print("fig ->", out)
