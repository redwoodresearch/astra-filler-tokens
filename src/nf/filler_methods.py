"""Are different filler methods equivalent at the same token count? Paired McNemar tests per task for every pair of
methods that share problems and exact filler-token counts (Astra, prompted no-CoT), pooled over depths and doses.

Methods (arm): dots in the user turn (XB), model-emitted dots (XC), statement-prefix/copies in the user turn (PB),
model-emitted prefix (PC), counting in the user turn (CB), model-emitted counting (CC), token-matched dots for the copy
arms (DB / DC), whole-statement copies (B with k>0 / C). Usage: uv run -m nf.filler_methods"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .analyze import ROOT, load

TAGS = [
    "dots_tasks",
    "prefix_tasks",
    "position_astra",
    "main",
    "fillerD_astra",
    "dots_dose",
    "greenblatt",
    "greenblatt2",
    "nhop",
]
METHOD = {
    "XB": "dots (user)",
    "XC": "dots (model)",
    "PB": "prefix (user)",
    "PC": "prefix (model)",
    "CB": "counting (user)",
    "CC": "counting (model)",
    "DB": "dots≈copies (user)",
    "DC": "dots≈copies (model)",
    "B": "copies (user)",
    "C": "copies (model)",
}


def main():
    df = pd.concat([load(t).assign(tag=t) for t in TAGS]).reset_index(drop=True)
    df = df[
        (df.model_eff == "gpt-6-astra:low")
        & (df.status == "completed")
        & (df.reasoning_tokens.fillna(0) == 0)
        & df.arm.isin(METHOD)
    ]
    df = df[
        (df.arm != "B") | (df.k > 0)
    ]  # B with k copies is the whole-statement-copies method; the k=0 baseline is excluded
    df["tokens"] = df.get("filler_tokens", pd.Series(index=df.index)).fillna(df.k).astype(float)
    # copy arms store k = number of copies; the token-matched dot arms DB/DC were built to pair with them by k
    copy_arms = df.arm.isin(["B", "C", "DB", "DC"])
    df.loc[copy_arms, "tokens"] = -df.loc[copy_arms, "k"].astype(float)
    df["method"] = df.arm.map(METHOD)
    df = df.drop_duplicates(["task", "problem_id", "method", "tokens"])
    rows = []
    methods = sorted(df.method.unique())
    for task, d in df.groupby("task"):
        for i, a in enumerate(methods):
            for b in methods[i + 1 :]:
                da, db = d[d.method == a], d[d.method == b]
                if da.empty or db.empty:
                    continue
                x = da[["problem_id", "tokens", "correct"]].merge(
                    db[["problem_id", "tokens", "correct"]], on="problem_id", suffixes=("_a", "_b")
                )
                same_sign = np.sign(x.tokens_a) == np.sign(x.tokens_b)
                close = (
                    np.abs(np.log(np.abs(x.tokens_a)) - np.log(np.abs(x.tokens_b))) < 0.1
                )  # within 10% (prefix cells round to line boundaries)
                x = x[same_sign & ((x.tokens_a < 0) & (x.tokens_a == x.tokens_b) | (x.tokens_a > 0) & close)]
                x = (
                    x.sort_values("tokens_a")
                    .drop_duplicates(["problem_id", "tokens_a"])
                    .drop_duplicates(["problem_id", "tokens_b"])
                )
                if len(x) < 30:
                    continue
                ca, cb = x.correct_a.astype(bool), x.correct_b.astype(bool)
                w, lo = int((ca & ~cb).sum()), int((cb & ~ca).sum())
                lv = sorted(x.tokens_a.abs().astype(int).unique())
                rows.append(
                    {
                        "task": task,
                        "method A": a,
                        "method B": b,
                        "n pairs": len(x),
                        "levels": (", ".join(map(str, lv[:8])) + (" …" if len(lv) > 8 else ""))
                        + (" copies" if x.tokens_a.iloc[0] < 0 else " tok"),
                        "acc A": ca.mean(),
                        "acc B": cb.mean(),
                        "A−B": ca.mean() - cb.mean(),
                        "A>B / B>A": f"{w}/{lo}",
                        "p": binomtest(w, w + lo).pvalue if w + lo else 1.0,
                    }
                )
    res = pd.DataFrame(rows).sort_values(["task", "method A", "method B"])
    pd.set_option("display.width", 250, "display.max_colwidth", 40)
    print(res.round(3).to_string(index=False))
    out = ROOT / "results" / "other" / "filler_methods.csv"
    res.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
