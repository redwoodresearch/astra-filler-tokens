"""Markdown tables of Astra's HLE and LiveBench results at the finest available granularity.

Conditions: no filler (B), counting 300 / 1000 (CB), reasoning allowed at effort low (R); paired gain and win/lose at
counting 1000. HLE uses the LLM-judge metric (string-match in a separate column at category level). Writes
results/bench_hard/figs/tables.md. Usage: uv run -m nf.bench_tables"""

import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.stats import binomtest

from .analyze import load, read_jsonl, tag_dir
from .tasks import dataset_row

COND = [("B", 0, "no filler"), ("CB", 300, "count 300"), ("CB", 1000, "count 1000"), ("R", 0, "reasoning low")]


def _load():
    df = pd.concat([load("bench_hard"), load("reasoning_low")]).reset_index(drop=True)
    df = df[
        (df.model_eff == "gpt-6-astra:low")
        & df.task.isin(["hle", "livebench"])
        & df.status.isin(["completed", "incomplete"])
    ].copy()
    df.loc[df.status == "incomplete", "correct"] = False
    df = df[(df.arm == "R") | (df.reasoning_tokens.fillna(0) == 0)]
    df = df[[(a, k) in {(c[0], c[1]) for c in COND} for a, k in zip(df.arm, df.k)]]
    judge = tag_dir("bench_hard") / "aux" / "hle_judge.jsonl"
    j = {(d["problem_id"], d["arm"], d["k"]): d["judge_correct"] for d in read_jsonl(judge)}
    df["judge"] = [(False if r.status == "incomplete" else j.get((r.problem_id, r.arm, r.k))) for r in df.itertuples()]
    df["idx"] = df.problem_id.str.extract(r"_i(\d+)_")[0].astype(int)
    return df.drop_duplicates(["problem_id", "arm", "k"])


def _table(df, group_cols, metric, title, min_n=1):
    lines = [
        f"### {title}",
        "",
        f"| {' | '.join(group_cols)} | n | no filler | count 300 | count 1000 | reasoning low | gain @1000 | win/lose | p |",
        "|" + "---|" * (len(group_cols) + 8),
    ]
    for key, d in df.groupby(group_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        b = d[(d.arm == "B")].set_index("problem_id")[metric].astype(float)
        if len(b) < min_n:
            continue
        accs = []
        for arm, k, _ in COND:
            s = d[(d.arm == arm) & (d.k == k)][metric].astype(float)
            accs.append(f"{s.mean():.2f}" if len(s) else "–")
        s = d[(d.arm == "CB") & (d.k == 1000)].set_index("problem_id")[metric].astype(float)
        c = b.index.intersection(s.index)
        w, lo = int(((s[c] == 1) & (b[c] == 0)).sum()), int(((b[c] == 1) & (s[c] == 0)).sum())
        p = binomtest(w, w + lo).pvalue if w + lo else 1.0
        lines.append(
            f"| {' | '.join(str(x) for x in key)} | {len(b)} | {' | '.join(accs)} | {s[c].mean() - b[c].mean():+.2f} | {w}/{lo} | {p:.3g} |"
        )
    return "\n".join(lines) + "\n"


def main():
    df = _load()
    out = [
        "# Astra (prompted no-CoT) on HLE and LiveBench: full breakdown",
        "",
        "Conditions: no filler; counting filler 1..300 (~600 tok) / 1..1000 (~2,000 tok); reasoning allowed at effort low. Gain and win/lose are paired per problem at counting 1000; p is an exact McNemar test.",
        "",
    ]
    h = df[df.task == "hle"].copy()
    meta = [dataset_row("hle", i) for i in h.idx]
    h["category"] = [m["category"] for m in meta]
    h["answer type"] = [m["answer_type"] for m in meta]
    h["subject"] = [m["subject"] for m in meta]
    out += [
        "## HLE (text-only, 2,158 questions)",
        "",
        "Metric: LLM judge (official HLE grading). String-match numbers for the category level follow.",
        "",
    ]
    h["all"] = "HLE all"
    out.append(_table(h, ["all"], "judge", "Overall"))
    out.append(_table(h, ["category"], "judge", "By category (judge)"))
    out.append(_table(h, ["category"], "correct", "By category (string match)"))
    out.append(_table(h, ["category", "answer type"], "judge", "By category × answer type (judge)"))
    out.append(_table(h, ["subject"], "judge", "By raw subject, n ≥ 15 (judge)", min_n=15))
    lb = df[df.task == "livebench"].copy()
    meta = [dataset_row("livebench", i) for i in lb.idx]
    lb["category"] = [m["source"].split("/")[1] for m in meta]
    lb["task"] = [m["task"] for m in meta]
    lb["subtask"] = [m.get("subtask") or "" for m in meta]
    # reasoning "level" (zebra grid size etc.) from the HF parquet
    r = pd.read_parquet(hf_hub_download("livebench/reasoning", "data/test-00000-of-00001.parquet", repo_type="dataset"))
    lvl = {q: int(v) for q, v in zip(r.question_id, pd.to_numeric(r["level"], errors="coerce")) if pd.notna(v)}
    ids = [m["id"] for m in meta]
    lb["subtask"] = [st or (f"level {lvl[i]:02d}" if i in lvl else "") for st, i in zip(lb.subtask, ids)]
    lb["all"] = "LiveBench all"
    out += [
        "## LiveBench (public 2024 releases, 618 answer-format questions)",
        "",
        "Metric: programmatic grade (LiveBench convention; AMPS_Hard with symbolic equivalence).",
        "",
    ]
    out.append(_table(lb, ["all"], "correct", "Overall"))
    out.append(_table(lb, ["category"], "correct", "By category"))
    out.append(_table(lb, ["category", "task"], "correct", "By task"))
    out.append(
        _table(
            lb[lb.subtask != ""], ["category", "task", "subtask"], "correct", "By subtask (where LiveBench defines one)"
        )
    )
    path = tag_dir("bench_hard") / "figs" / "tables.md"
    path.write_text("\n".join(out))
    print("\n".join(out))
    print("wrote", path)


if __name__ == "__main__":
    main()
