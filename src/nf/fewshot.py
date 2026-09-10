"""DeepSeek V3.2 (reasoning off): zero-shot vs 5-shot, with and without counting filler, on AIME-Plus-Plus and Gen-Arithmetic.
Compared with the zero-shot Astra / Opus / Sol cells already in results/greenblatt(2). Usage: uv run -m nf.fewshot"""

import pandas as pd
from scipy.stats import binomtest

from .analyze import aimepp_tier, load, tag_dir

MODELS = [
    "gpt-6-astra:low",
    "claude-opus-4-5-20251101:off",
    "claude-opus-5:off",
    "gpt-5.6-sol:none",
    "deepseek/deepseek-v3.2:off",
]


def _cells(df, label):
    rows = []
    for me, dm in df.groupby("model_eff"):
        for (task, dep), d in dm.groupby(["task", "depth"]):
            b = d[d.arm == "B"].set_index("problem_id").correct.astype(bool)
            if b.empty:
                continue
            rec = {"model": me, "prompt": label, "task": task, "depth": dep, "n": len(b), "no_filler": b.mean()}
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
    return rows


def _prep(df):
    df = df[df.status.isin(["completed"]) & df.task.isin(["aimepp", "arith"]) & df.model_eff.isin(MODELS)]
    ant = df.get("served_provider", pd.Series(index=df.index, dtype=object)).eq("anthropic")
    rt = df.reasoning_tokens.fillna(0)
    df = df[((rt <= 10) & ant) | ((rt == 0) & ~ant)]
    df = df[((df.arm == "B") & (df.k == 0)) | ((df.arm == "CB") & df.k.isin([300, 1000]))].copy()
    # AIME-Plus-Plus: AIME tier as depth "AIME", all tiers as depth "all"
    a = df[df.task == "aimepp"].copy()
    a["depth"] = a.problem_id.map(aimepp_tier)
    a2 = df[df.task == "aimepp"].copy()
    a2["depth"] = "all"
    return pd.concat([df[df.task == "arith"], a[a.depth == "AIME"], a2])


def main():
    zero = _prep(pd.concat([load("greenblatt"), load("greenblatt2")]).reset_index(drop=True))
    few = _prep(load("fewshot"))
    res = pd.DataFrame(_cells(zero, "0-shot") + _cells(few, "5-shot"))
    res["depth"] = res.depth.astype(str)
    res = res.sort_values(["task", "depth", "model", "prompt"])
    pd.set_option("display.width", 220, "display.max_rows", 200)
    print(res.round(3).to_string(index=False))
    out = tag_dir("fewshot") / "figs"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "fewshot.csv", index=False)


if __name__ == "__main__":
    main()
