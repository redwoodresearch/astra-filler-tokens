"""LiveBench math by subtask (AMC/AIME contests, AMPS_Hard problem types, olympiad USAMO/IMO): no-CoT, filler, reasoning-low.
Usage: uv run -m nf.lb_math_split"""

import pandas as pd

from .analyze import load, tag_dir
from .tasks import dataset_row


def main():
    df = pd.concat([load("bench_hard"), load("reasoning_low")])
    df = df[
        (df.model_eff == "gpt-6-astra:low") & (df.task == "livebench") & df.status.isin(["completed", "incomplete"])
    ].copy()
    df.loc[df.status == "incomplete", "correct"] = False
    df = df[(df.arm == "R") | (df.reasoning_tokens.fillna(0) == 0)]
    meta = [dataset_row("livebench", int(p.split("_i")[1].split("_")[0])) for p in df.problem_id]
    df["task2"] = [m["task"] for m in meta]
    df["subtask"] = [m.get("subtask") or m["task"] for m in meta]
    df = df[df.task2.isin(["math_comp", "AMPS_Hard", "olympiad"])]
    df["cond"] = df.arm + "@" + df.k.astype(str)
    out = []
    for (t2, st), d in df.groupby(["task2", "subtask"]):
        b = d[d.cond == "B@0"].set_index("problem_id").correct.astype(bool)
        rec = {"task": t2, "subtask": st, "n": len(b), "no_cot": b.mean()}
        for c, lab in [("CB@300", "filler300"), ("CB@1000", "filler1000"), ("R@0", "reasoning_low")]:
            s = d[d.cond == c].set_index("problem_id").correct.astype(bool)
            rec[lab] = s.mean()
            if c == "CB@1000":
                cm = b.index.intersection(s.index)
                rec["win/lose@1000"] = f"{int((s[cm] & ~b[cm]).sum())}/{int((b[cm] & ~s[cm]).sum())}"
        out.append(rec)
    res = pd.DataFrame(out).sort_values(["task", "n"], ascending=[False, False])
    pd.set_option("display.width", 200)
    print(res.round(2).to_string(index=False))
    res.to_csv(tag_dir("bench_hard") / "figs" / "lb_math_split.csv", index=False)


if __name__ == "__main__":
    main()
