"""Position ablation on Astra: where does filler have to sit?

Conditions (all matched by token count where possible), on nhop (2-7 hops), Gen-Arithmetic 15 ops, AIME/HMMT:
  after the statement, user turn   : XB (dots), CB (counting)         -- tags greenblatt / nhop
  before the statement, user turn  : YB (dots)                        -- tag position_astra
  model-emitted after empty reasoning : XC (dots), CC (counting)      -- tag position_astra
Prints paired gains vs the shared no-filler baseline (exact McNemar p) and draws a 1x3 figure."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .analyze import load, tag_dir

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
MODEL = "gpt-6-astra:low"
PANELS = {"nhop": "n-hop facts (3-5 hops pooled)", "arith": "Gen-Arithmetic, 15 ops", "aime": "AIME/HMMT 2024-26"}
# (arm, k) -> (label, group, color)
COND = {
    ("XB", 300): ("300 dots", "after, user", "#4878CF"),
    ("YB", 300): ("300 dots", "before, user", "#B47CC7"),
    ("XC", 300): ("300 dots", "model-emitted", "#D65F5F"),
    ("CB", 300): ("count 1..300", "after, user", "#4878CF"),
    ("CC", 300): ("count 1..300", "model-emitted", "#D65F5F"),
    ("CB", 1000): ("count 1..1000", "after, user", "#4878CF"),
    ("CC", 1000): ("count 1..1000", "model-emitted", "#D65F5F"),
    ("YB", 2000): ("2000 dots", "before, user", "#B47CC7"),
}
XORDER = ["300 dots", "count 1..300", "count 1..1000", "2000 dots"]
GROUPS = {"after, user": -0.25, "before, user": 0.0, "model-emitted": 0.25}


def main():
    df = pd.concat([load(t) for t in ("greenblatt", "nhop", "position_astra")]).reset_index(drop=True)
    df = df[(df.model_eff == MODEL) & (df.status == "completed") & (df.reasoning_tokens.fillna(0) == 0)]
    df = df[df.task.isin(PANELS) & ((df.task != "arith") | (df.depth == 15))]
    df = df[(df.arm == "B") & (df.k == 0) | df.apply(lambda r: (r.arm, r.k) in COND, axis=1)]
    df = df.drop_duplicates(["task", "problem_id", "arm", "k"])
    out = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, (task, title) in zip(axes, PANELS.items()):
        d = df[df.task == task]
        if task == "nhop":
            d = d[d.depth.isin([3, 4, 5])]
        base = d[d.arm == "B"].set_index("problem_id").correct.astype(bool)
        b = base.mean()
        ax.axhline(b, color="gray", ls=":", lw=1.2, label=f"no filler ({b:.2f}, n={len(base)})")
        seen = set()
        for (arm, k), (lab, grp, col) in COND.items():
            s = d[(d.arm == arm) & (d.k == k)].set_index("problem_id").correct.astype(bool)
            idx = base.index.intersection(s.index)
            if len(idx) == 0:
                continue
            bb, t = base[idx], s[idx]
            win, lose = int((t & ~bb).sum()), int((bb & ~t).sum())
            p = binomtest(win, win + lose, 0.5).pvalue if win + lose else 1.0
            out.append(
                {
                    "task": task,
                    "cond": lab,
                    "where": grp,
                    "n": len(idx),
                    "base": bb.mean(),
                    "acc": t.mean(),
                    "gain": t.mean() - bb.mean(),
                    "win": win,
                    "lose": lose,
                    "p": p,
                }
            )
            x = XORDER.index(lab) + GROUPS[grp]
            ax.errorbar(
                x,
                t.mean(),
                yerr=1.96 * np.sqrt(t.mean() * (1 - t.mean()) / len(t)),
                fmt="o",
                color=col,
                capsize=3,
                ms=6,
                label=grp if grp not in seen else None,
            )
            seen.add(grp)
        ax.set_xticks(range(len(XORDER)), XORDER)
        ax.set(title=title, ylim=(-0.02, 1.02), xlim=(-0.6, len(XORDER) - 0.4))
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("accuracy, no CoT")
    fig.suptitle(
        f"{MODEL} (prompted no-CoT): filler position — after statement (user) vs before statement (user) vs model-emitted after empty reasoning",
        fontsize=11,
    )
    fig.savefig(tag_dir("position_astra") / "position_astra.png", dpi=150)
    res = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    res.to_csv(tag_dir("position_astra") / "position_gains.csv", index=False)
    # per-hop detail for nhop
    d = df[df.task == "nhop"]
    tab = (
        d.assign(cond=np.where(d.arm == "B", "no filler", d.arm + "@" + d.k.astype(str)))
        .groupby(["cond", "depth"])
        .correct.mean()
        .unstack("depth")
    )
    print("\nn-hop accuracy by hops:\n", tab.round(2).to_string())
    print("wrote results/position_astra/position_astra.png, position_gains.csv")


if __name__ == "__main__":
    main()
