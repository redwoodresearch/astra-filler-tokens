"""Build data/aimepp.jsonl from the ulamai/AIME-Plus-Plus dataset on HuggingFace (all four tiers, in tier order).
Needed only to run new evaluations; the analyses use data/aimepp_meta.jsonl. Usage: uv run scripts/build_aimepp.py"""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parents[1] / "data"
TIERS = ["aime", "aime-hard", "aime-graduate", "aime-researcher"]


def main():
    rows = []
    for tier in TIERS:
        path = hf_hub_download("ulamai/AIME-Plus-Plus", f"data/{tier}.jsonl", repo_type="dataset")
        for line in Path(path).open():  # noqa: SIM115
            r = json.loads(line)
            rows.append(
                {
                    "id": r["id"],
                    "source": f"ulamai/AIME-Plus-Plus {r['tier']}",
                    "statement": r["problem"],
                    "answer": str(r["answer"]),
                    "answer_spec": "the final integer answer only (digits, no units, no commas)",
                    "chance": 0.0,
                    "tier": r["tier"],
                }
            )
    (OUT / "aimepp.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print("aimepp:", len(rows), {t: sum(r["tier"] == t for r in rows) for t in dict.fromkeys(r["tier"] for r in rows)})


if __name__ == "__main__":
    main()
