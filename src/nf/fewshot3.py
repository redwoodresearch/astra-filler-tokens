"""3-shot test for the non-Astra models: demonstrations (held-out problems, correct answers) carry the same filler as the
query (500 dots or none). Compared with the zero-shot cells: no filler vs ~600 counting tokens (the nearest dose).
Gen-Arithmetic 5-7 ops (pooled, n=300) and n-hop 2 / 3 hops (n=147). Usage: uv run -m nf.fewshot3"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .analyze import load, tag_dir

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
MODELS = {
    "gpt-5.6-sol:none": "gpt-5.6-sol",
    "claude-opus-5:off": "claude-opus-5",
    "claude-opus-4-5-20251101:off": "claude-opus-4.5",
    "deepseek/deepseek-v3.2:off": "deepseek-v3.2",
}
CELLS = [
    ("arith", (5, 6, 7), "Gen-Arithmetic 5-7 ops"),
    ("nhop", (2,), "n-hop, 2 hops"),
    ("nhop", (3,), "n-hop, 3 hops"),
]


def _valid(df):
    df = df[(df.status == "completed") & df.model_eff.isin(MODELS)]
    ant = df.get("served_provider", pd.Series(index=df.index, dtype=object)).eq("anthropic")
    rt = df.reasoning_tokens.fillna(0)
    return df[ant | ((rt == 0) & ~ant)]


def _pair(d, base_sel, filler_sel):
    b = d[base_sel(d)].set_index("problem_id").correct.astype(bool)
    f = d[filler_sel(d)].set_index("problem_id").correct.astype(bool)
    c = b.index.intersection(f.index)
    if len(c) == 0:
        return None
    w, lo = int((f[c] & ~b[c]).sum()), int((b[c] & ~f[c]).sum())
    return {
        "n": len(c),
        "no_filler": b[c].mean(),
        "filler": f[c].mean(),
        "gain": f[c].mean() - b[c].mean(),
        "win": w,
        "lose": lo,
        "p": binomtest(w, w + lo).pvalue if w + lo else 1.0,
    }


def main():
    zero = _valid(pd.concat([load("greenblatt"), load("nhop")]).reset_index(drop=True))
    few = _valid(load("fewshot3"))
    rows = []
    for me, lab in MODELS.items():
        for task, depths, cname in CELLS:
            z = zero[(zero.model_eff == me) & (zero.task == task) & zero.depth.isin(depths)]
            f = few[(few.model_eff == me) & (few.task == task) & few.depth.isin(depths)]
            r0 = _pair(z, lambda d: (d.arm == "B") & (d.k == 0), lambda d: (d.arm == "CB") & (d.k == 300))
            r3 = _pair(f, lambda d: (d.arm == "B") & (d.k == 0), lambda d: (d.arm == "XB") & (d.k == 500))
            for prompt, r in [("0-shot (filler = counting 300, ~600 tok)", r0), ("3-shot (filler = 500 dots)", r3)]:
                if r:
                    rows.append({"model": lab, "cell": cname, "prompt": prompt, **r})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    out = tag_dir("fewshot3") / "figs"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "fewshot3.csv", index=False)
    fig, axes = plt.subplots(1, len(CELLS), figsize=(4.6 * len(CELLS), 4.2), constrained_layout=True, sharey=True)
    for ax, (_, _, cname) in zip(axes, CELLS):
        d = res[res.cell == cname]
        labs = list(MODELS.values())
        x = np.arange(len(labs))
        for i, (prompt, col_b, col_f) in enumerate(
            [("0-shot", "#BBBBBB", "#6ACC65"), ("3-shot", "#777777", "#2C8C2C")]
        ):
            dd = d[d.prompt.str.startswith(prompt)].set_index("model").reindex(labs)
            ax.bar(x + (i * 2 - 1.5) * 0.2, dd.no_filler, 0.2, color=col_b, label=f"{prompt}, no filler")
            ax.bar(x + (i * 2 - 0.5) * 0.2, dd.filler, 0.2, color=col_f, label=f"{prompt}, ~500-600 filler tokens")
        ax.set_xticks(x, labs, fontsize=8, rotation=15)
        ax.set(title=cname, ylim=(0, 1))
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("accuracy, no CoT")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Does 3-shot prompting unlock a filler effect in the non-Astra models?")
    fig.savefig(out / "fewshot3.png", dpi=150)
    print("wrote", out / "fewshot3.png")


if __name__ == "__main__":
    main()
