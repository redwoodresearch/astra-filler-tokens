"""No-CoT baseline vs 2,000 filler tokens vs reasoning allowed at effort low, per task, Astra.

Sources: results/greenblatt (arith, aime: arms B / CB@1000), results/bench_hard (hle, livebench: B / CB@1000),
results/reasoning_low (arm R). Prints accuracy per condition with hidden-reasoning and total-output token statistics for R,
and draws a grouped bar chart. HLE rows also under the LLM judge when results/bench_hard/aux/hle_judge.jsonl covers arm R."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import aimepp_tier, jsonl_exists, load, read_jsonl, tag_dir
from .tasks import dataset_row

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
ROWS = [
    ("arith", 15, "Gen-Arithmetic 15 ops"),
    ("arith", 10, "Gen-Arithmetic 10 ops"),
    ("aimepp", "AIME", "AIME-Plus-Plus\nAIME tier"),
    ("aimepp", "all", "AIME-Plus-Plus\nall 157"),
    ("livebench", 1, "LiveBench"),
    ("hle", 1, "HLE"),
]
COND = [
    ("B", 0, "no CoT, no filler", "gray"),
    ("CB", 1000, "no CoT, counting 1..1000", "#2C8C2C"),
    ("R", 0, "reasoning allowed (effort low)", "#D65F5F"),
]


def _lb_sub(pid):
    return dataset_row("livebench", int(pid.split("_i")[1].split("_")[0]))["task"]


def main():
    df = pd.concat([load(t) for t in ("greenblatt", "greenblatt2", "bench_hard", "reasoning_low")]).reset_index(
        drop=True
    )
    df = df[(df.model_eff == "gpt-6-astra:low") & df.status.isin(["completed", "incomplete"])]
    df.loc[df.status == "incomplete", "correct"] = False
    df = df[(df.arm == "R") | (df.reasoning_tokens.fillna(0) == 0)]  # no-CoT arms must have no hidden reasoning
    judge = tag_dir("bench_hard") / "aux" / "hle_judge.jsonl"
    jd = (
        {(d["problem_id"], d["arm"], d["k"]): d["judge_correct"] for d in read_jsonl(judge)}
        if jsonl_exists(judge)
        else {}
    )
    out = []
    for task, depth, label in ROWS + [
        ("livebench", s, f"LB {s}")
        for s in ("math_comp", "AMPS_Hard", "olympiad", "zebra_puzzle", "spatial", "web_of_lies_v2", "cta")
    ]:
        d = df[df.task == task]
        if task == "livebench" and isinstance(depth, str):
            d = d[d.problem_id.map(_lb_sub) == depth]
        elif task == "aimepp":
            d = d if depth == "all" else d[d.problem_id.map(aimepp_tier) == depth]
        else:
            d = d[d.depth == depth]
        for arm, k, cname, _ in COND:
            s = d[(d.arm == arm) & (d.k == k)].drop_duplicates("problem_id")
            if s.empty:
                continue
            rec = {"task": label, "cond": cname, "n": len(s), "acc": s.correct.astype(float).mean()}
            if task == "hle" and jd:
                jv = [jd.get((p, arm, k)) for p in s.problem_id]
                rec["acc_judge"] = np.mean([bool(v) for v in jv]) if any(v is not None for v in jv) else np.nan
            if arm == "R":
                rt, ot = s.reasoning_tokens.fillna(0), s.output_tokens
                rec.update(
                    reasoning_median=rt.median(),
                    reasoning_mean=rt.mean(),
                    reasoning_p90=rt.quantile(0.9),
                    output_mean=ot.mean(),
                    incomplete=(s.status == "incomplete").mean(),
                )
            out.append(rec)
    res = pd.DataFrame(out)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(res.round(3).to_string(index=False))
    fig_dir = tag_dir("reasoning_low") / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(fig_dir / "reasoning_low.csv", index=False)
    main_ = res[~res.task.str.startswith("LB ")]
    tasks = list(dict.fromkeys(main_.task))
    fig, ax = plt.subplots(figsize=(1.9 * len(tasks) + 2.5, 4.3), constrained_layout=True)
    x = np.arange(len(tasks))
    for i, (_, _, cname, col) in enumerate(COND):
        vals, ns = [], []
        for t in tasks:
            r = main_[(main_.task == t) & (main_.cond == cname)]
            # HLE bars use the LLM-judge metric (official HLE grading); everything else the programmatic grade
            vals.append(
                (
                    r.acc_judge.iloc[0]
                    if t == "HLE" and "acc_judge" in r and pd.notna(r.acc_judge.iloc[0])
                    else r.acc.iloc[0]
                )
                if len(r)
                else np.nan
            )
            ns.append(r.n.iloc[0] if len(r) else 1)
        vals, ns = np.array(vals), np.array(ns)
        ax.bar(
            x + (i - 1) * 0.27,
            vals,
            0.27,
            color=col,
            yerr=1.96 * np.sqrt(vals * (1 - vals) / ns),
            capsize=2,
            label=cname,
        )
    ax.set_xticks(x, [f"{t}\n(n={main_[main_.task == t].n.max()})" for t in tasks], fontsize=9)
    ax.set(ylabel="accuracy", ylim=(0, 1.02), title="gpt-6-astra: prompted no-CoT vs filler vs reasoning at effort low")
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.set_title("gpt-6-astra: prompted no-CoT vs filler vs reasoning at effort low", pad=28)
    fig.savefig(fig_dir / "reasoning_low.png", dpi=150)
    print("wrote", fig_dir / "reasoning_low.png")


if __name__ == "__main__":
    main()
