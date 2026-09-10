"""Non-toy benchmarks (GSM8K, GPQA Diamond, MMLU-Pro STEM): accuracy vs filler for six filler types.
usage: uv run -m nf.bench bench_dots bench_filler2"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import load, tag_dir

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
MODELS = ["gpt-4o:default", "gpt-5.6-sol:none", "claude-fable-5-1:low", "gpt-6-astra:low"]
BENCH = {"gsm8k": "GSM8K (numeric)", "gpqa": "GPQA Diamond (4-way)", "mmlupro": "MMLU-Pro STEM (10-way)"}
ARMS = {
    "XB": ("dots, user", "#4878CF", "-", "o"),
    "XC": ("dots, model", "#4878CF", "--", "s"),
    "CB": ("counting 1..N, user", "#6ACC65", "-", "o"),
    "CC": ("counting 1..N, model", "#6ACC65", "--", "s"),
    "RB": ("question repeats, user", "#D65F5F", "-", "o"),
    "C": ("question repeats, model", "#D65F5F", "--", "s"),
}


def se(p, n):
    p, n = np.asarray(p, float), np.asarray(n, float)
    return 1.96 * np.sqrt(p * (1 - p) / n)


if __name__ == "__main__":
    df = pd.concat([load(t) for t in sys.argv[1:]]).reset_index(drop=True)
    df = df[(df.status != "error") & df.model_eff.isin(MODELS) & df.task.isin(BENCH)]
    df = df[(df.arm != "B") | (df.k == 0)]
    # Fable's model-emitted-repeat arm is a cap artefact (Anthropic tokenizer counts benchmark prose ~1.3x o200k, so
    # the o200k-based cap truncated 221/600 calls); excluded rather than shown as a spurious drop.
    df = df[~((df.model_eff == "claude-fable-5-1:low") & (df.arm == "C"))]
    ft = df.get("filler_tokens", df.k).fillna(df.k)
    ft = np.where(
        (df.arm == "C") & (ft <= 0), df.expected_output_tokens.fillna(0) - 4, ft
    )  # arm C did not record filler_tokens
    df["x"] = np.where(df.arm == "B", 0, np.maximum(ft, 1))
    t = (
        df.groupby(["model_eff", "task", "arm", "k"])
        .agg(
            n=("correct", "size"),
            acc=("correct", "mean"),
            x=("x", "mean"),
            chance=("chance", "first"),
            parse=("parsed", lambda v: v.notna().mean()),
            compliant=("compliant", "mean"),
            rt=("reasoning_tokens", lambda v: v.fillna(0).max()),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(
        len(BENCH),
        len(MODELS),
        figsize=(4.6 * len(MODELS), 4.2 * len(BENCH)),
        constrained_layout=True,
        squeeze=False,
        sharey="row",
    )
    rows = []
    for r, (task, tname) in enumerate(BENCH.items()):
        for c, me in enumerate(MODELS):
            ax = axes[r, c]
            d = t[(t.model_eff == me) & (t.task == task)]
            base = d[d.arm == "B"]
            b = base.acc.mean() if len(base) else np.nan
            nb = int(base.n.sum()) if len(base) else 0
            ax.axhline(b, color="gray", ls=":", lw=1.2, label=f"no filler ({b:.2f}, n={nb})")
            if len(d):
                ax.axhline(d.chance.iloc[0], color="gray", lw=0.8, ls="-.")
            for arm, (lab, col, ls, mk) in ARMS.items():
                s = d[(d.arm == arm) & (d.k > 0)].sort_values("x")
                if s.empty:
                    continue
                ax.errorbar(
                    s.x, s.acc, yerr=se(s.acc, s.n), fmt=mk, ls=ls, color=col, lw=1.6, capsize=2, ms=4, label=lab
                )
                for _, row in s.iterrows():
                    rows.append(
                        {
                            "model_eff": me,
                            "task": task,
                            "arm": arm,
                            "k": row.k,
                            "filler_tokens": round(row.x),
                            "n": row.n,
                            "acc": row.acc,
                            "baseline": b,
                            "gain": row.acc - b,
                            "gain_se": np.sqrt(row.acc * (1 - row.acc) / row.n + (b * (1 - b) / max(nb, 1))),
                            "parse": row.parse,
                            "compliant": row.compliant,
                            "max_reasoning_tokens": row.rt,
                        }
                    )
            ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.05)
            if r == 0:
                ax.set_title(me, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{tname}\naccuracy, no CoT (↑)", fontsize=10)
            if r == len(BENCH) - 1:
                ax.set_xlabel("filler length (tokens), log", fontsize=10)
            ax.legend(frameon=False, fontsize=6.5, loc="best")
    fig.suptitle(
        "Non-toy benchmarks: no-CoT accuracy vs filler length for six filler types (n=100 questions per point; dotted = no filler; dash-dot = chance)",
        fontsize=11,
    )
    out = tag_dir("bench_dots") / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "bench_filler.png", dpi=170)
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(R.round(3).to_string(index=False))
    R.to_csv(out / "bench_filler.csv", index=False)
    print("\nbaselines (no filler):")
    print(t[t.arm == "B"].groupby(["task", "model_eff"]).agg(acc=("acc", "mean"), n=("n", "sum")).round(3).to_string())
    print("fig ->", out / "bench_filler.png")
