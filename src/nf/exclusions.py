"""Audit of excluded calls for the post's figures: per result tag and model, how many stored rows were dropped and why
(API error, refusal, truncated output, hidden reasoning tokens), and which dose cells the figures drop because fewer
than half the problems were answered. Usage: uv run -m nf.exclusions"""

import pandas as pd

from .analyze import load

TAGS = {
    "nhop": "N-hop",
    "greenblatt": "AIME/HMMT + Gen-Arithmetic",
    "greenblatt2": "AIME-Plus-Plus",
    "bench_hard": "HLE + LiveBench",
    "reasoning_low": "reasoning low",
    "position_astra": "position ablation",
}
MODELS = [
    "gpt-6-astra:low",
    "gpt-5.6-sol:none",
    "claude-opus-5:off",
    "claude-opus-4-5-20251101:off",
    "deepseek/deepseek-v3.2:off",
]


def main():
    rows, dropped_cells = [], []
    for tag, label in TAGS.items():
        df = load(tag)
        df = df[df.model_eff.isin(MODELS)]
        ant = df.get("served_provider", pd.Series(index=df.index, dtype=object)).eq("anthropic")
        rt = df.reasoning_tokens.fillna(0)
        hidden = (
            (rt > 0) & ~ant & (df.arm != "R")
        )  # Anthropic rows run with thinking disabled: no hidden reasoning possible
        for me, d in df.groupby("model_eff"):
            h = hidden[d.index]
            rec = {
                "tag": label,
                "model": me.split(":")[0],
                "rows": len(d),
                "error": int((d.status == "error").sum()),
                "refusal": int((d.status == "refusal").sum()),
                "truncated": int((d.status == "incomplete").sum()),
                "hidden reasoning": int((h & (d.status == "completed")).sum()),
            }
            rec["excluded"] = (
                rec["error"]
                + rec["refusal"]
                + rec["hidden reasoning"]
                + (0 if tag == "bench_hard" else rec["truncated"])
            )
            rec["excluded %"] = 100 * rec["excluded"] / max(1, len(d))
            rows.append(rec)
            # dose cells the figures drop (fewer than half the problems answered)
            ok = d[(d.status == "completed") & ~h & d.arm.isin(["B", "XB", "CB"])]
            for (task, depth, arm, k), g in ok.groupby(["task", "depth", "arm", "k"]):
                n_all = len(d[(d.task == task) & (d.depth == depth) & (d.arm == arm) & (d.k == k)])
                if len(g) < 0.5 * n_all:
                    dropped_cells.append(
                        {
                            "tag": label,
                            "model": me.split(":")[0],
                            "task": task,
                            "depth": depth,
                            "arm": arm,
                            "k": k,
                            "answered": len(g),
                            "of": n_all,
                        }
                    )
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(res.round(2).to_string(index=False))
    print(
        "\nTotal rows:",
        res.rows.sum(),
        "| excluded:",
        res.excluded.sum(),
        f"({100 * res.excluded.sum() / res.rows.sum():.2f}%)",
    )
    print("\nDose cells dropped from figures (fewer than half the problems answered):")
    print(pd.DataFrame(dropped_cells).to_string(index=False) if dropped_cells else "  none")


if __name__ == "__main__":
    main()
