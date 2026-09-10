"""Parse-rate confound audit for the dose runs: is accuracy vs filler length driven by parse failures?
Per model x arm x filler length (pooled over tasks/depths): parse rate (an ANSWER line was found), strict compliance
(exact filler + single answer line), and accuracy over all / parsed-only / compliant-only calls.
usage: uv run -m nf.parse_audit <tags...>"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import load, tag_dir
from .dots_dose import to_dot_counts

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
ARMS = {
    "B": "no filler",
    "XB": "dots after (user)",
    "XC": "dots (model)",
    "YB": "dots before (user)",
    "PB": "repetition (user)",
    "PC": "repetition (model)",
}

if __name__ == "__main__":
    df = pd.concat([load(t) for t in sys.argv[1:]])
    df = df[(df.status != "error") & df.arm.isin(["B", "XB", "XC", "YB", "PB", "PC", "DB", "DC"])]
    df = df[(df.arm != "B") | (df.k == 0)]
    df = to_dot_counts(df).reset_index(
        drop=True
    )  # concat left duplicate indices; the .loc lookups below need unique ones
    df["parsed_ok"] = df.parsed.notna()
    ft = df.get("filler_tokens", df.k)
    ft = ft.fillna(df.k)
    df["x"] = np.where(
        df.arm == "B", 0, np.where(ft > 0, ft, df.k)
    )  # X arms recorded filler_tokens=0 before the fix; k is the dot count
    g = (
        df.groupby(["model_eff", "arm", "x"])
        .agg(
            n=("correct", "size"),
            parse_rate=("parsed_ok", "mean"),
            compliant=("compliant", "mean"),
            acc_all=("correct", "mean"),
            acc_parsed=("correct", lambda v: df.loc[v.index][df.loc[v.index].parsed_ok].correct.mean()),
            acc_compliant=("correct", lambda v: df.loc[v.index][df.loc[v.index].compliant].correct.mean()),
        )
        .reset_index()
    )
    models = sorted(df.model_eff.unique())
    fig, axes = plt.subplots(2, len(models), figsize=(4.6 * len(models), 8), constrained_layout=True, squeeze=False)
    cols = {"B": "gray", "XB": "#4878CF", "XC": "#D65F5F", "YB": "#B47CC7", "PB": "#6ACC65", "PC": "#C4AD66"}
    cell = (
        df.groupby(["model_eff", "task", "depth", "arm", "x"])
        .agg(
            n=("correct", "size"),
            parse_rate=("parsed_ok", "mean"),
            acc_all=("correct", "mean"),
            acc_parsed=("correct", lambda v: df.loc[v.index][df.loc[v.index].parsed_ok].correct.mean()),
        )
        .reset_index()
    )
    for c, me in enumerate(models):
        d = g[g.model_eff == me]
        ax = axes[0, c]
        for arm, lab in ARMS.items():
            s = d[d.arm == arm].sort_values("x")
            if s.empty:
                continue
            ax.plot(s.x.clip(lower=0.7), s.parse_rate, "o-", color=cols[arm], label=f"{lab}: parse rate")
            ax.plot(
                s.x.clip(lower=0.7),
                s.compliant,
                "s--",
                color=cols[arm],
                ms=4,
                alpha=0.6,
                label=f"{lab}: strict compliance",
            )
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(f"{me}\nparse rate / strict compliance vs filler length", fontsize=9)
        ax.set_xlabel("filler length (tokens), log; k=0 at 0.7")
        ax.legend(frameon=False, fontsize=6)
        ax = axes[1, c]
        cc = cell[cell.model_eff == me]
        for arm, lab in ARMS.items():
            s = cc[cc.arm == arm]
            if s.empty:
                continue
            ax.scatter(s.acc_all, s.acc_parsed, s=14, color=cols[arm], alpha=0.7, label=lab)
        ax.plot([0, 1], [0, 1], "k-", lw=0.8)
        ax.set_xlim(-0.03, 1.05)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("accuracy over all calls (unparsed = wrong)")
        ax.set_ylabel("accuracy over parsed calls only")
        ax.set_title("per cell (task x depth x arm x length): parsed-only vs all", fontsize=9)
        ax.legend(frameon=False, fontsize=6)
    out = tag_dir("horizon") / "figs" / "parse_audit.png"
    fig.savefig(out, dpi=170)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    summ = df.groupby(["model_eff", "arm"]).agg(
        n=("correct", "size"),
        parse_rate=("parsed_ok", "mean"),
        compliant=("compliant", "mean"),
        acc_all=("correct", "mean"),
        acc_parsed=("correct", lambda v: df.loc[v.index][df.loc[v.index].parsed_ok].correct.mean()),
        min_parse_rate_any_x=("parsed_ok", lambda v: df.loc[v.index].groupby("x").parsed_ok.mean().min()),
    )
    print(summ.round(3).to_string())
    # cells where parse failures could matter: parse rate < 0.9 and |acc_all - acc_parsed| > 0.05
    flag = g[(g.parse_rate < 0.9) & ((g.acc_all - g.acc_parsed).abs() > 0.05)]
    print("\ncells (model, arm, filler length) with parse rate < 0.9 AND accuracy shifted > 0.05 by exclusion:")
    print(flag.round(3).to_string(index=False) if len(flag) else "  none")
    g.to_csv(tag_dir("horizon") / "parse_audit_cells.csv", index=False)
    print("fig ->", out)
