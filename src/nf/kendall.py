"""Kendall tau-b trend test: dot count (incl. 0) vs per-call correctness, per model x task x depth x arm.
usage: uv run -m nf.kendall <tags...>"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .analyze import load, tag_dir
from .dots_dose import to_dot_counts

if __name__ == "__main__":
    df = pd.concat([load(t) for t in sys.argv[1:]]).reset_index(drop=True)
    df = df[(df.status != "error") & df.arm.isin(["B", "XB", "XC", "DB", "DC"]) & (df.task != "ca")]
    df = df[(df.arm != "B") | (df.k == 0)]
    rt = df.reasoning_tokens.fillna(0)
    ant = df.get("served_provider", pd.Series(index=df.index, dtype=object)).eq("anthropic")
    df = df[(df.status != "refusal") & (ant | (rt == 0) & ~ant)]
    df = df[~((df.model_eff == "claude-opus-5:off") & (df.arm.isin(["XC", "DC"])))]  # API refused this arm
    df = to_dot_counts(df)
    df["arm"] = df.arm.map({"DB": "XB", "DC": "XC"}).fillna(df.arm)
    rows = []
    for (me, task, depth, arm), g in df[df.arm != "B"].groupby(["model_eff", "task", "depth", "arm"]):
        base = df[(df.model_eff == me) & (df.task == task) & (df.depth == depth) & (df.arm == "B")]
        cell = pd.concat([base, g])
        if cell.k.nunique() < 3 or len(cell) < 60:
            continue
        tau, p = kendalltau(cell.k.values, cell.correct.astype(int).values)
        rows.append(
            {
                "model_eff": me,
                "task": task,
                "depth": depth,
                "arm": arm,
                "n_calls": len(cell),
                "n_levels": cell.k.nunique(),
                "max_dots": int(cell.k.max()),
                "acc_k0": base.correct.mean() if len(base) else np.nan,
                "acc_maxk": g[g.k == g.k.max()].correct.mean(),
                "tau_b": tau,
                "p": p,
            }
        )
    R = pd.DataFrame(rows).sort_values(["model_eff", "task", "depth", "arm"])
    R["sig"] = np.select([R.p < 0.001, R.p < 0.01, R.p < 0.05], ["***", "**", "*"], "")
    out = tag_dir("horizon") / "kendall_dots_trend.csv"
    R.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(R.round(3).to_string(index=False))
    print(
        "\n=== per model: cells, cells with positive trend p<0.01, cells with negative trend p<0.01, median tau (user-supplied arm) ==="
    )
    u = R[R.arm == "XB"]
    print(
        u.groupby("model_eff")
        .agg(
            cells=("tau_b", "size"),
            pos_sig=("tau_b", lambda t: int(((t > 0) & (u.loc[t.index].p < 0.01)).sum())),
            neg_sig=("tau_b", lambda t: int(((t < 0) & (u.loc[t.index].p < 0.01)).sum())),
            median_tau=("tau_b", "median"),
            max_tau=("tau_b", "max"),
        )
        .round(3)
        .to_string()
    )
    print("\n=== per model x task (user-supplied arm): tau per depth ===")
    print(u.pivot_table(index=["model_eff", "task"], columns="depth", values="tau_b").round(2).to_string())
    print("csv ->", out)
