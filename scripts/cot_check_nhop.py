"""With-CoT sanity check of the n-hop extension: Astra with reasoning allowed (effort medium, 20k cap), 20 problems each at 5/6/7 hops.
Tells whether the no-CoT floor at 6-7 hops is a knowledge/solvability floor or a serial-compute floor. 60 calls."""

import asyncio
import json

from openai import AsyncOpenAI

from nf.keys import openai_key
from nf.nhop_grade import check_answer
from nf.tasks import make_problem

out = open("results/nhop/aux/cot_check_astra.jsonl", "a")  # noqa: SIM115


async def one(client, hops, i):
    p = make_problem(hops, i, task="nhop")
    r = await client.responses.create(
        model="gpt-6-astra",
        reasoning={"effort": "medium"},
        max_output_tokens=20000,
        store=False,
        input=[
            {
                "role": "user",
                "content": p.statement + "\n\nSolve the question. Finish with a final line `ANSWER: <answer>`.",
            }
        ],
    )
    txt = r.output_text or ""
    ans = txt.rsplit("ANSWER:", 1)[-1].strip().splitlines()[0].strip() if "ANSWER:" in txt else None
    rec = {
        "hops": hops,
        "idx": i,
        "truth": p.answer,
        "parsed": ans,
        "correct": bool(ans) and check_answer(ans, p.answer),
        "status": r.status,
        "reasoning_tokens": r.usage.output_tokens_details.reasoning_tokens,
        "output_tokens": r.usage.output_tokens,
        "tail": txt[-300:],
    }
    out.write(json.dumps(rec) + "\n")
    out.flush()
    return rec


async def main():
    client = AsyncOpenAI(api_key=openai_key(), timeout=900)
    recs = await asyncio.gather(*(one(client, h, i) for h in (5, 6, 7) for i in range(20)))
    import pandas as pd

    df = pd.DataFrame(recs)
    print(
        df.groupby("hops")
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
    print(df[["hops", "truth", "parsed", "correct", "reasoning_tokens"]].to_string(index=False))


asyncio.run(main())
