"""Framing ablation: dots described as "carry no information; ignore them" (XB) vs "extra space for you to process the
problem" (XI), same problems and token counts, paired McNemar. Usage: uv run -m nf.framing"""

import pandas as pd
from scipy.stats import binomtest

from .analyze import ROOT, aimepp_tier, load


def main():
    df = pd.concat([load(t) for t in ("nhop", "greenblatt", "greenblatt2")]).reset_index(drop=True)
    df = df[
        (df.status == "completed")
        & (df.reasoning_tokens.fillna(0) == 0)
        & df.arm.isin(["XB", "XI"])
        & df.k.isin([256, 1024])
    ]
    df = df[df.task.isin(["nhop", "arith", "aime", "aimepp"])].copy()
    df["cell"] = df.task + df.depth.astype(str).where(df.task.isin(["nhop", "arith"]), "")
    df.loc[df.task == "aimepp", "cell"] = "aimepp " + df[df.task == "aimepp"].problem_id.map(aimepp_tier)
    rows = []
    for (me, cell, k), d in df.groupby(["model_eff", "cell", "k"]):
        x = d[d.arm == "XB"].set_index("problem_id").correct.astype(bool)
        y = d[d.arm == "XI"].set_index("problem_id").correct.astype(bool)
        c = x.index.intersection(y.index)
        if len(c) < 20:
            continue
        w, lo = int((y[c] & ~x[c]).sum()), int((x[c] & ~y[c]).sum())
        rows.append(
            {
                "model": me,
                "cell": cell,
                "dots": k,
                "n": len(c),
                "ignore-them": x[c].mean(),
                "thinking-space": y[c].mean(),
                "Δ": y[c].mean() - x[c].mean(),
                "space>ignore / ignore>space": f"{w}/{lo}",
                "p": binomtest(w, w + lo).pvalue if w + lo else 1.0,
            }
        )
    res = pd.DataFrame(rows).sort_values(["model", "cell", "dots"])
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    # pooled per model
    for me, d in df.groupby("model_eff"):
        x = d[d.arm == "XB"].set_index(["problem_id", "k"]).correct.astype(bool)
        y = d[d.arm == "XI"].set_index(["problem_id", "k"]).correct.astype(bool)
        c = x.index.intersection(y.index)
        w, lo = int((y[c] & ~x[c]).sum()), int((x[c] & ~y[c]).sum())
        print(
            f"pooled {me}: n={len(c)} ignore-them {x[c].mean():.3f} thinking-space {y[c].mean():.3f} Δ={y[c].mean() - x[c].mean():+.3f} ({w}/{lo}, p={binomtest(w, w + lo).pvalue:.3g})"
        )
    res.to_csv(ROOT / "results" / "other" / "framing.csv", index=False)


if __name__ == "__main__":
    main()
