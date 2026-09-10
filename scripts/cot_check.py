"""With-CoT sanity check of the AIME-Plus-Plus tiers: Astra with reasoning allowed (effort medium, 20k token cap)."""

import asyncio
import json

from openai import AsyncOpenAI

from nf.keys import openai_key
from nf.prompts import _norm_answer
from nf.tasks import make_problem

TIERS = [json.loads(l)["tier"] for l in open("data/aimepp.jsonl")]  # noqa: SIM115
idx = {
    "AIME Hard": [i for i, t in enumerate(TIERS) if t == "AIME Hard"][:20],
    "AIME": [i for i, t in enumerate(TIERS) if t == "AIME"][:10],
}
out = open("results/greenblatt2/aux/cot_check_astra.jsonl", "a")  # noqa: SIM115


async def one(client, i):
    p = make_problem(1, i, task="aimepp")
    r = await client.responses.create(
        model="gpt-6-astra",
        reasoning={"effort": "medium"},
        max_output_tokens=20000,
        store=False,
        input=[
            {
                "role": "user",
                "content": p.statement + "\n\nSolve the problem. Finish with a final line `ANSWER: <integer>`.",
            }
        ],
    )
    txt = r.output_text or ""
    ans = _norm_answer(txt.rsplit("ANSWER:", 1)[-1].strip().split()[0]) if "ANSWER:" in txt else None
    rec = {
        "idx": i,
        "tier": TIERS[i],
        "truth": p.answer,
        "parsed": ans,
        "correct": ans == _norm_answer(p.answer),
        "status": r.status,
        "reasoning_tokens": r.usage.output_tokens_details.reasoning_tokens,
        "output_tokens": r.usage.output_tokens,
        "tail": txt[-200:],
    }
    out.write(json.dumps(rec) + "\n")
    out.flush()
    return rec


async def main():
    client = AsyncOpenAI(api_key=openai_key(), timeout=900)
    recs = await asyncio.gather(*(one(client, i) for tier in idx for i in idx[tier]))
    import pandas as pd

    df = pd.DataFrame(recs)
    print(
        df.groupby("tier")
        .agg(
            n=("correct", "size"),
            acc=("correct", "mean"),
            incomplete=("status", lambda s: (s != "completed").mean()),
            reasoning_tok_median=("reasoning_tokens", "median"),
            reasoning_tok_max=("reasoning_tokens", "max"),
        )
        .round(3)
        .to_string()
    )
    print(df[["tier", "truth", "parsed", "correct", "status", "reasoning_tokens"]].to_string(index=False))


asyncio.run(main())
