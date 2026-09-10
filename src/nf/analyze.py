"""Summarize a results/<tag> directory: accuracy by model x depth x arm x k, compliance, reasoning tokens."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def tag_dir(tag: str) -> Path:
    """Directory holding a result tag: results/main/<tag> (post figures), results/other/<tag> (further experiments),
    or results/<tag> (new runs). Returns the first that exists, else results/<tag>."""
    for d in (RESULTS / "main" / tag, RESULTS / "other" / tag, RESULTS / tag):
        if d.exists():
            return d
    return RESULTS / tag


CHANCE = {
    "perm": 1 / 8,
    "lookup": 0.1,
    "modchain": 1 / 7,
    "reach": 0.5,
    "ca": 0.5,
    "stack": 1 / 9,
    "xor": 0.5,
}  # pre-"chance"-field records


def read_jsonl(path):
    """Iterate JSON rows of path or path + '.gz' (result logs are stored gzipped)."""
    import gzip

    path = Path(path)
    if not path.exists() and path.with_name(path.name + ".gz").exists():
        path = path.with_name(path.name + ".gz")
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as f:
        for line in f:
            yield json.loads(line)


def jsonl_exists(path) -> bool:
    path = Path(path)
    return path.exists() or path.with_name(path.name + ".gz").exists()


def load(tag: str) -> pd.DataFrame:
    rows = []
    d = tag_dir(tag)
    for f in sorted(list(d.glob("*.jsonl")) + list(d.glob("*.jsonl.gz"))):
        for r in read_jsonl(f):
            r.pop("request", None)
            r.pop("response", None)
            rows.append(r)
    df = pd.DataFrame(rows)
    if "task" not in df:
        df["task"] = None
    df["task"] = df["task"].fillna(df.problem_id.map(lambda p: "perm" if p.startswith("d") else p.split("_d")[0]))
    if "chance" not in df:
        df["chance"] = None
    df["chance"] = df["chance"].astype(float).fillna(df.task.map(CHANCE))
    df["cond"] = df["arm"] + df["k"].astype(str)
    df["model_eff"] = df["model"] + ":" + df["effort"].fillna("default").astype(str)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df.status != "error"]
    g = ok.groupby(["model_eff", "depth", "arm", "k"])
    s = g.agg(
        n=("correct", "size"),
        acc=("correct", "mean"),
        compliant=("compliant", "mean"),
        acc_compliant=("correct", lambda x: ok.loc[x.index].query("compliant")["correct"].mean()),
        incomplete=("status", lambda x: (x == "incomplete").mean()),
        reason_tok=("reasoning_tokens", "mean"),
        out_tok=("output_tokens", "mean"),
        latency=("latency_s", "median"),
    ).reset_index()
    return s


if __name__ == "__main__":
    tag = sys.argv[1]
    df = load(tag)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print(f"{len(df)} records, {int((df.status == 'error').sum())} errors")
    s = summarize(df)
    print(s.round(3).to_string(index=False))
    s.to_csv(ROOT / "results" / tag / "summary.csv", index=False)
    # Headline: C_k - B_k accuracy gap per model x depth x k.
    piv = s.pivot_table(index=["model_eff", "depth", "k"], columns="arm", values="acc")
    if {"B", "C"} <= set(piv.columns):
        piv["C_minus_B"] = piv["C"] - piv["B"]
        print("\nC minus B (model-emitted minus user-supplied filler):")
        print(piv.round(3).to_string())


def aimepp_tier(problem_id: str) -> str:
    """Tier label of an AIME-Plus-Plus problem id (AIME / AIME Hard / AIME-Graduate / AIME-Researcher)."""
    from .tasks import dataset_row

    return dataset_row("aimepp", int(problem_id.split("_i")[1].split("_")[0]))["tier"]
