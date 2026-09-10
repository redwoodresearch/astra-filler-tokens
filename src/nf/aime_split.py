"""AIME/HMMT filler gains split by competition family, source contest, and position within contest (difficulty proxy).
Existing data only (results/greenblatt, Astra). Usage: uv run -m nf.aime_split"""

import pandas as pd
from scipy.stats import binomtest

from .analyze import load, tag_dir
from .tasks import dataset_row


def _gains(d, name):
    base = d[d.arm == "B"].set_index("problem_id").correct.astype(bool)
    out = []
    for k in (300, 1000):
        s = d[(d.arm == "CB") & (d.k == k)].set_index("problem_id").correct.astype(bool)
        c = base.index.intersection(s.index)
        b, t = base[c], s[c]
        w, lo = int((t & ~b).sum()), int((b & ~t).sum())
        out.append(
            {
                "split": name,
                "k": k,
                "n": len(c),
                "base": b.mean(),
                "filler": t.mean(),
                "gain": t.mean() - b.mean(),
                "win": w,
                "lose": lo,
                "p": binomtest(w, w + lo).pvalue if w + lo else 1.0,
            }
        )
    return out


def main():
    df = load("greenblatt")
    df = df[
        (df.model_eff == "gpt-6-astra:low")
        & (df.task == "aime")
        & (df.status == "completed")
        & (df.reasoning_tokens.fillna(0) == 0)
    ]
    df = df[((df.arm == "B") & (df.k == 0)) | ((df.arm == "CB") & df.k.isin([300, 1000]))].copy()
    idx = df.problem_id.str.extract(r"_i(\d+)_")[0].astype(int)
    meta = pd.DataFrame([dataset_row("aime", i) for i in idx])
    df["source"] = meta.source.values
    df["num"] = meta.id.str.extract(r"_(\d+)$")[0].astype(int).values
    df["family"] = df.source.str.extract(r"^(aime|hmmt|brumo|cmimc|smt)")[0].str.upper()
    df["pos"] = df.groupby("source").num.transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["early", "middle", "late"])
    )
    rows = []
    for fam, d in df.groupby("family"):
        rows += _gains(d, f"family {fam}")
    for src, d in df.groupby("source"):
        rows += _gains(d, f"  {src}")
    for pos, d in df.groupby("pos", observed=True):
        rows += _gains(d, f"contest position {pos}")
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    res.to_csv(tag_dir("greenblatt") / "figs" / "aime_split_gains.csv", index=False)


if __name__ == "__main__":
    main()
