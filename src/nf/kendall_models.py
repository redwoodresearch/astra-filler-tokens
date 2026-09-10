"""Kendall tau-b between filler tokens and per-call correctness, per model and dataset, over the same cells the dose
figures use (no-filler baseline at 0 plus the dot doses; counting cells only where a model has no dot sweep; cells the
model mostly refused dropped). Usage: uv run -m nf.kendall_models"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .analyze import ROOT, aimepp_tier, load
from .ideal import MODELS, _valid


def _cells(d):
    """Same selection as ideal._curve, returned as per-call rows."""
    cells = d.groupby(["arm", "x"]).correct.agg(["mean", "size"]).reset_index()
    cells = cells[cells["size"] >= 0.5 * cells["size"].max()]
    xb = cells[cells.arm == "XB"].x.to_numpy()
    keep = cells.apply(lambda r: r.arm != "CB" or not any(abs(np.log2(r.x / v)) < 1 for v in xb if v > 0), axis=1)
    sel = set(zip(cells[keep].arm, cells[keep].x))
    return d[[(a, x) in sel for a, x in zip(d.arm, d.x)]]


def main():
    nh = _valid(load("nhop"))
    gb = _valid(pd.concat([load("greenblatt"), load("greenblatt2")]).reset_index(drop=True))
    sets = {
        "Gen-Arithmetic 15 ops": gb[(gb.task == "arith") & (gb.depth == 15)],
        "n-hop, 2 hops": nh[(nh.task == "nhop") & (nh.depth == 2)],
        "n-hop, 3 hops": nh[(nh.task == "nhop") & (nh.depth == 3)],
        "n-hop, 4 hops": nh[(nh.task == "nhop") & (nh.depth == 4)],
        "n-hop, 2-7 hops pooled": nh[nh.task == "nhop"],
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
    res.to_csv(ROOT / "results" / "main" / "kendall_models.csv", index=False)


if __name__ == "__main__":
    main()
