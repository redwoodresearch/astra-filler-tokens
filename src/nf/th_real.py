"""Rough no-CoT time horizons for Astra at 0 / ~600 / ~2,000 filler tokens on the real-task sets (AIME/HMMT, AIME-Plus-Plus,
HLE, LiveBench) plus n-hop and Gen-Arithmetic, using eyeballed human solve-time ratings (minutes) listed in RATINGS below.
Chance-corrected logistic P = c + (1-c) * sigmoid(a + b * log2(minutes)) fit by weighted MLE (benchmarks weighted equally,
1/(B * N_b) per problem as in Think Fast), 1,000 problem-level bootstraps for CIs. HLE uses the LLM-judge metric.
Usage: uv run -m nf.th_real"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .analyze import ROOT, aimepp_tier, load, read_jsonl, tag_dir
from .tasks import dataset_row

# ---- human solve-time ratings (minutes), eyeballed; edit here to re-fit ----
RATINGS = {
    "aime": {
        "aime": (4, 24),
        "hmmt": (3, 15),
        "other": (3, 12),
    },  # (first problem, last problem) by contest position, linear
    "aimepp": {"AIME": 12, "AIME Hard": 40, "AIME-Graduate": 90, "AIME-Researcher": 180},
    "hle": {
        "Math": 30,
        "Computer Science/AI": 20,
        "Physics": 20,
        "Chemistry": 15,
        "Biology/Medicine": 15,
        "Engineering": 20,
        "Humanities/Social Science": 10,
        "Other": 10,
    },
    "livebench": {
        "math_comp": 5,
        "AMPS_Hard": 3,
        "olympiad": 15,
        "zebra_puzzle": 8,
        "spatial": 2,
        "web_of_lies_v2": 2,
        "cta": 0.5,
    },
    "nhop": {h: 1.0 * h for h in range(2, 8)},  # about a minute per fact lookup for a person with a search engine
    "arith": {5: 1.5, 6: 1.5, 7: 1.5, 10: 2.5, 15: 4},
}
COND = {"no filler": ("B", 0), "~600 filler tokens": ("CB", 300), "~2,000 filler tokens": ("CB", 1000)}


def minutes(r) -> float:
    if r.task == "aime":
        src = dataset_row("aime", r.idx)
        fam = "aime" if src["source"].startswith("aime") else "hmmt" if src["source"].startswith("hmmt") else "other"
        n = int(src["id"].rsplit("_", 1)[1])
        n_max = 15 if fam == "aime" else 10
        lo, hi = RATINGS["aime"][fam]
        return lo + (hi - lo) * (min(n, n_max) - 1) / (n_max - 1)
    if r.task == "aimepp":
        return RATINGS["aimepp"][aimepp_tier(r.problem_id)]
    if r.task == "hle":
        return RATINGS["hle"][dataset_row("hle", r.idx)["category"]]
    if r.task == "livebench":
        return RATINGS["livebench"][dataset_row("livebench", r.idx)["task"]]
    return RATINGS[r.task][int(r.depth)]


def fit(x, y, c, w):
    def nll(p):
        pr = c + (1 - c) / (1 + np.exp(-(p[0] + p[1] * x)))
        pr = np.clip(pr, 1e-6, 1 - 1e-6)
        return -np.sum(w * (y * np.log(pr) + (1 - y) * np.log(1 - pr))) + 0.01 * p[1] ** 2

    return minimize(nll, [0.0, -1.0], method="Nelder-Mead").x


def horizons(a, b):
    th50 = 2 ** (-a / b)
    th80 = 2 ** ((np.log(0.8 / 0.2) - a) / b)
    return th50, th80


def main(exclude=()):  # exclude: task names or "aimepp-hard" to drop the harder AIME-Plus-Plus tiers (sensitivity)
    df = pd.concat([load(t) for t in ("greenblatt", "greenblatt2", "bench_hard", "nhop")]).reset_index(drop=True)
    df = df[
        (df.model_eff == "gpt-6-astra:low")
        & df.status.isin(["completed", "incomplete"])
        & (df.reasoning_tokens.fillna(0) == 0)
    ]
    df.loc[df.status == "incomplete", "correct"] = False
    df = df[df.task.isin(["aime", "aimepp", "hle", "livebench", "nhop", "arith"])].copy()
    df["idx"] = df.problem_id.str.extract(r"_i(\d+)_")[0].astype(int)
    df = df[~df.task.isin(exclude)]
    if "aimepp-hard" in exclude:
        keep = df.task != "aimepp"
        keep[df.task == "aimepp"] = df[df.task == "aimepp"].problem_id.map(aimepp_tier) == "AIME"
        df = df[keep]
    judge = {
        (d["problem_id"], d["arm"], d["k"]): d["judge_correct"]
        for d in read_jsonl(tag_dir("bench_hard") / "aux" / "hle_judge.jsonl")
    }
    hle = df.task == "hle"
    if hle.any():
        df.loc[hle, "correct"] = [
            bool(judge.get((p, a, k), False)) for p, a, k in zip(df[hle].problem_id, df[hle].arm, df[hle].k)
        ]
    df["minutes"] = [minutes(r) for r in df.itertuples()]
    rows = []
    rng = np.random.default_rng(0)
    for cond, (arm, k) in COND.items():
        d = df[(df.arm == arm) & (df.k == k)].drop_duplicates("problem_id")
        nb = d.groupby("task").size()
        w = (1.0 / (len(nb) * d.task.map(nb))).to_numpy()
        x, y, c = np.log2(d.minutes.to_numpy()), d.correct.astype(float).to_numpy(), d.chance.astype(float).to_numpy()
        a, b = fit(x, y, c, w)
        th50, th80 = horizons(a, b)
        boots = []
        for _ in range(1000):
            i = rng.integers(0, len(d), len(d))
            boots.append(horizons(*fit(x[i], y[i], c[i], w[i])))
        boots = np.array(boots)
        rows.append(
            {
                "condition": cond,
                "n": len(d),
                "TH50 min": th50,
                "TH50 lo": np.percentile(boots[:, 0], 2.5),
                "TH50 hi": np.percentile(boots[:, 0], 97.5),
                "TH80 min": th80,
                "TH80 lo": np.percentile(boots[:, 1], 2.5),
                "TH80 hi": np.percentile(boots[:, 1], 97.5),
                "slope": b,
            }
        )
        # per-benchmark accuracy by rating bucket, for eyeballing
        if cond == "no filler":
            print("accuracy by benchmark and human minutes (no filler / ~600 / ~2,000):")
        for (task, m), g in d.groupby(["task", "minutes"]):
            others = [
                df[(df.arm == aa) & (df.k == kk) & (df.task == task) & (df.minutes == m)].correct.mean()
                for aa, kk in COND.values()
            ]
            if cond == "no filler":
                print(f"  {task:10s} {m:6.1f} min  n={len(g):4d}  " + " / ".join(f"{v:.2f}" for v in others))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n", res.round(2).to_string(index=False))
    res.to_csv(ROOT / "results" / "other" / "th_real.csv", index=False)


if __name__ == "__main__":
    main()
