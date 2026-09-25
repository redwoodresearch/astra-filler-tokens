"""Kendall tau-b between filler tokens and per-call correctness, per model and dataset, over the same cells the dose
figures use (no-filler baseline at 0 plus the dot doses; counting cells only where a model has no dot sweep; cells the
model mostly refused dropped). Usage: uv run -m nf.kendall_models [fewshot10]
(fewshot10: the same test over the 10-shot runs, post models only; writes
results/other/ideal_fewshot10/kendall_models_fewshot10.csv)"""

import math

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .analyze import ROOT, aimepp_tier, load
from .ideal import MODELS, _valid


def _cells(d):
    """Same selection as ideal._curve, returned as per-call rows."""
    cells = d.groupby(["arm", "x"]).correct.agg(["mean", "size"]).reset_index()
    if cells.empty:
        return d
    cells = cells[cells["size"] >= 0.5 * cells["size"].max()]
    xb = cells[cells.arm == "XB"].x.to_numpy()
    keep = cells.apply(lambda r: r.arm != "CB" or not any(abs(np.log2(r.x / v)) < 1 for v in xb if v > 0), axis=1)
    sel = set(zip(cells[keep].arm, cells[keep].x))
    return d[[(a, x) in sel for a, x in zip(d.arm, d.x)]]


def main(fewshot10: bool = False):
    if fewshot10:  # Astra to 4,096 dots, others to 1,024; Opus 5 not run few-shot
        nh = gb = _valid(pd.concat([load("fewshot10_astra"), load("fewshot10_others")]).reset_index(drop=True))
    else:
        nh = _valid(load("nhop"))
        gb = _valid(pd.concat([load("greenblatt"), load("greenblatt2")]).reset_index(drop=True))
    sets = {
        "Gen-Arithmetic 15 ops": gb[(gb.task == "arith") & (gb.depth == 15)],
        "n-hop, 2 hops": nh[(nh.task == "nhop") & (nh.depth == 2)],
        "n-hop, 3 hops": nh[(nh.task == "nhop") & (nh.depth == 3)],
        "n-hop, 4 hops": nh[(nh.task == "nhop") & (nh.depth == 4)],
        ("n-hop, 2+4 hops pooled" if fewshot10 else "n-hop, 2-7 hops pooled"): nh[nh.task == "nhop"],
        "AIME-Plus-Plus, AIME tier": gb[gb.task == "aimepp"][lambda d: d.problem_id.map(aimepp_tier) == "AIME"],
        "AIME/HMMT 2024-26": gb[gb.task == "aime"],
    }
    rows = []
    for sname, d0 in sets.items():
        for me, (lab, _) in MODELS.items():
            d = _cells(d0[d0.model_eff == me])
            if d.empty:
                continue
            tau, p = kendalltau(np.log1p(d.x.to_numpy(dtype=float)), d.correct.to_numpy(dtype=float))
            rows.append(
                {
                    "dataset": sname,
                    "model": lab,
                    "n calls": len(d),
                    "doses": d.x.nunique(),
                    "acc at 0": d[d.x == 0].correct.mean(),
                    "acc at max": d[d.x == d.x.max()].correct.mean(),
                    "tau": tau,
                    "p": p,
                }
            )
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    out = (
        ROOT
        / "results"
        / ("other/ideal_fewshot10/kendall_models_fewshot10.csv" if fewshot10 else "main/kendall_models.csv")
    )
    res.to_csv(out, index=False)
    out.with_suffix(".md").write_text(post_table(res, fewshot10))
    print(post_table(res, fewshot10))


POST_COLS = [
    ("gpt-6-astra", "Astra"),
    ("gpt-5.6-sol", "5.6-Sol"),
    ("claude-opus-5", "Opus 5"),
    ("claude-opus-4.5", "Opus 4.5"),
    ("deepseek-v3.2", "Deepseek"),
]
POST_ROWS = [
    "Gen-Arithmetic 15 ops",
    "n-hop, 2 hops",
    "n-hop, 4 hops",
    "AIME-Plus-Plus, AIME tier",
    "AIME/HMMT 2024-26",
]


def post_table(res, fewshot10=False):
    """The post's appendix layout: one row per dataset, one column per model, `tau (p)`, bold p < 0.05."""
    cols = [c for c in POST_COLS if c[0] in set(res.model)]

    def p_str(p):
        if p < 1e-3:
            return f"p<1e{math.floor(math.log10(p)) + 1}"
        return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"

    def cell(r):
        if r.empty:
            return "—"
        r = r.iloc[0]
        t = f"{r.tau + 0.0:.2f}".replace("-0.00", "0.00").replace("-", "−")
        return (f"**{t}**" if r.p < 0.05 else t) + f" ({p_str(r.p)})"

    lines = ["| Dataset | " + " | ".join(c for _, c in cols) + " |", "|---|" + "---|" * len(cols)]
    for d in POST_ROWS:
        name = d.replace("n-hop", "N-hop").replace("2024-26", "2024–26")
        lines.append(
            f"| {name} | " + " | ".join(cell(res[(res.dataset == d) & (res.model == m)]) for m, _ in cols) + " |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import sys

    main("fewshot10" in sys.argv)
