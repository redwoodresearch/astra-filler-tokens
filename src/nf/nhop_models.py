"""Natural-fact n-hop across models: accuracy vs hops without filler and with counting filler 1..1000, plus paired gain.
Usage: uv run -m nf.nhop_models"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binomtest

from .analyze import load, tag_dir

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
LABELS = {
    "gpt-6-astra:low": "gpt-6-astra (prompted no-CoT)",
    "gpt-5.6-sol:none": "gpt-5.6-sol",
    "claude-opus-5:off": "claude-opus-5 (thinking off)",
    "claude-opus-4-5-20251101:off": "claude-opus-4.5 (thinking off)",
    "deepseek/deepseek-v3.2:off": "deepseek-v3.2 (reasoning off)",
}


def main():
    df = load("nhop")
    df = df[(df.task == "nhop") & (df.status == "completed") & (df.status != "refusal")]
    ant = df.get("served_provider", pd.Series(index=df.index, dtype=object)).eq("anthropic")
    rt = df.reasoning_tokens.fillna(0)
    df = df[((rt <= 10) & ant) | ((rt == 0) & ~ant)]
    df = df[((df.arm == "B") & (df.k == 0)) | ((df.arm == "CB") & df.k.isin([300, 1000]))]
    rows = []
    for me, dm in df.groupby("model_eff"):
        for h, d in dm.groupby("depth"):
            b = d[d.arm == "B"].set_index("problem_id").correct.astype(bool)
            rec = {"model": me, "hops": h, "n": len(b), "no_filler": b.mean()}
            for k in (300, 1000):
                s = d[(d.arm == "CB") & (d.k == k)].set_index("problem_id").correct.astype(bool)
                c = b.index.intersection(s.index)
                rec[f"count_{k}"] = s.mean()
                if k == 1000:
                    w, lo = int((s[c] & ~b[c]).sum()), int((b[c] & ~s[c]).sum())
                    rec.update(
                        gain=s[c].mean() - b[c].mean(), win=w, lose=lo, p=binomtest(w, w + lo).pvalue if w + lo else 1.0
                    )
            rows.append(rec)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    out = tag_dir("nhop") / "figs"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "nhop_models.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for me, g in res.groupby("model"):
        lab = LABELS.get(me, me)
        c = axes[0].plot(g.hops, g.no_filler, marker="o", label=lab)[0].get_color()
        axes[1].plot(g.hops, g.count_1000, marker="o", color=c, label=lab)
        axes[2].plot(g.hops, g.gain, marker="o", color=c, label=lab)
    axes[0].set(title="no filler", ylabel="accuracy, no CoT", ylim=(-0.02, 1.02))
    axes[1].set(title="counting filler 1..1000 (~2,000 tokens)", ylim=(-0.02, 1.02))
    axes[2].set(title="paired gain from filler", ylim=(-0.1, 0.6))
    axes[2].axhline(0, color="gray", lw=0.8)
    for ax in axes:
        ax.set_xlabel("hops")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Natural-fact n-hop questions, no-CoT, by model")
    fig.savefig(out / "nhop_models.png", dpi=150)
    print("wrote", out / "nhop_models.png")


if __name__ == "__main__":
    main()
