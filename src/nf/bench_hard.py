"""HLE and LiveBench: no-CoT accuracy with and without counting filler, per subset, paired per problem.

Usage: uv run -m nf.bench_hard [tag ...]   (default tag bench_hard). Uses the programmatic grade in `correct`; if
results/bench_hard/aux/hle_judge.jsonl exists (scripts/judge_hle.py), an HLE row is also reported under the judge."""

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .analyze import jsonl_exists, load, read_jsonl, tag_dir
from .tasks import dataset_row

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
COLORS = {0: "gray", 300: "#6ACC65", 1000: "#2C8C2C"}


def _subset(r):
    idx = int(r.problem_id.split("_i")[1].split("_")[0])
    if r.task == "hle":
        d = dataset_row("hle", idx)
        return [("HLE all", "hle"), (f"HLE {d['answer_type']}", "hle"), (f"HLE {d['category']}", "hle-cat")]
    d = dataset_row("livebench", idx)
    cat = d["source"].split("/")[1]  # math / reasoning / data_analysis
    return [("LiveBench all", "lb"), (f"LB {cat} all", "lb-cat"), (f"LB {d['task']}", "lb")]


LB_TREE = {  # category -> tasks, in display order
    "math": ["math_comp", "AMPS_Hard", "olympiad"],
    "reasoning": ["zebra_puzzle", "spatial", "web_of_lies_v2"],
    "data_analysis": ["cta"],
}


def main(tags):
    df = pd.concat([load(t) for t in tags]).reset_index(drop=True)
    df = df[
        df.status.isin(["completed", "incomplete"])
        & df.task.isin(["hle", "livebench"])
        & (df.reasoning_tokens.fillna(0) == 0)
    ]
    df.loc[df.status == "incomplete", "correct"] = (
        False  # answer ran past a cap sized from the true answer: wrong form, scored wrong
    )
    df = df[(df.arm == "B") & (df.k == 0) | (df.arm == "CB")]
    judge = tag_dir("bench_hard") / "aux" / "hle_judge.jsonl"
    if jsonl_exists(judge):
        j = {(d["problem_id"], d["arm"], d["k"], d["model_eff"]): d["judge_correct"] for d in read_jsonl(judge)}
        df["judge"] = [
            False if r.status == "incomplete" else j.get((r.problem_id, r.arm, r.k, r.model_eff))
            for r in df.itertuples()
        ]
    rows = []
    for me, dm in df.groupby("model_eff"):
        subs = {}
        for r in dm.itertuples():
            for name, grp in _subset(r):
                subs.setdefault((name, grp), []).append(r.Index)
        for (name, grp), idxs in subs.items():
            d = dm.loc[idxs]
            for metric in ["correct"] + (["judge"] if "judge" in d and d.judge.notna().any() and grp != "lb" else []):
                base = d[d.arm == "B"].set_index("problem_id")[metric].astype(float)
                for k, s in d[d.arm == "CB"].groupby("k"):
                    s = s.set_index("problem_id")[metric].astype(float)
                    common = base.index.intersection(s.index)
                    b, t = base[common], s[common]
                    if metric == "judge":
                        m = b.notna() & t.notna()
                        b, t = b[m], t[m]
                    win, lose = int(((t == 1) & (b == 0)).sum()), int(((b == 1) & (t == 0)).sum())
                    p = binomtest(win, win + lose, 0.5).pvalue if win + lose else 1.0
                    rows.append(
                        {
                            "model": me,
                            "subset": name,
                            "group": grp,
                            "metric": metric,
                            "k": k,
                            "n": len(b),
                            "base": b.mean(),
                            "filler": t.mean(),
                            "gain": t.mean() - b.mean(),
                            "win": win,
                            "lose": lose,
                            "p": p,
                        }
                    )
    res = pd.DataFrame(rows).sort_values(["model", "group", "subset", "metric", "k"])
    pd.set_option("display.width", 220, "display.max_rows", 500)
    print(res.round(3).to_string(index=False))
    out = tag_dir("bench_hard") / "figs"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "bench_hard_gains.csv", index=False)
    # figures: (a) headline HLE + LiveBench totals, (b) LiveBench subcategories; programmatic metric
    # figures: HLE subsets under the LLM judge (the official HLE metric), LiveBench under the programmatic grade
    main_ = res[
        ((res.subset.str.startswith("HLE")) & (res.metric == "judge"))
        | (~res.subset.str.startswith("HLE") & (res.metric == "correct"))
    ]
    for me, dm in main_.groupby("model"):
        cats = dm[dm.group == "hle-cat"].drop_duplicates("subset").sort_values("n", ascending=False).subset.tolist()
        panels = {
            "headline": ["HLE all", "LiveBench all"],
            "livebench": ["LiveBench all"] + [f"LB {c} all" for c in LB_TREE],
            "hle_categories": ["HLE all"] + cats,
        }
        for name, subsets in panels.items():
            fig, ax = plt.subplots(
                figsize=((1.15 if name == "hle_categories" else 1.7) * len(subsets) + 2.5, 4.2), constrained_layout=True
            )
            x = np.arange(len(subsets))
            for i, (k, col) in enumerate([(0, "gray"), (300, "#6ACC65"), (1000, "#2C8C2C")]):
                vals, ns = [], []
                for sname in subsets:
                    r = dm[(dm.subset == sname) & (dm.k == (k or 300))].iloc[0]
                    vals.append(r.base if k == 0 else r.filler)
                    ns.append(r.n)
                vals, ns = np.array(vals), np.array(ns)
                ax.bar(
                    x + (i - 1) * 0.27,
                    vals,
                    0.27,
                    color=col,
                    yerr=1.96 * np.sqrt(vals * (1 - vals) / ns),
                    capsize=2,
                    label="no filler" if k == 0 else f"counting 1..{k}",
                )
            labels = [
                (
                    s
                    if s in ("HLE all", "LiveBench all")
                    else s.replace("LB ", "")
                    .replace("HLE ", "")
                    .replace(" all", "")
                    .replace("/", "/\n")
                    .replace("data_analysis", "data analysis")
                )
                for s in subsets
            ]
            ax.set_xticks(x, labels, fontsize=9)
            ax.set(ylabel="accuracy", ylim=(0, 1.02))
            short = me.split(":")[0].replace("gpt-6-astra", "Astra")
            ax.set_title(
                {
                    "headline": f"HLE and LiveBench performance (no-CoT {short})",
                    "livebench": f"LiveBench performance (no-CoT {short})",
                    "hle_categories": f"HLE performance (no-CoT {short})",
                }[name],
                fontsize=11,
            )
            ax.legend(frameon=False, fontsize=8, loc="upper left")
            path = out / f"bench_hard_{name}_{me.replace(':', '_')}.png"
            fig.savefig(path, dpi=150)
            print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1:] or ["bench_hard"])
