"""Probe accepted reasoning efforts + reasoning_tokens under the empty-CoT instruction (uncapped)."""

import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

from nf.keys import openai_key
from nf.prompts import build, check
from nf.tasks import make_problem

COMBOS = [(m, e) for m in ("gpt-6-astra", "gpt-5.6-sol") for e in ("none", "minimal", "low")]
CASES = [("B", 0), ("C", 2)]
out = Path("results/probe_effort.jsonl")


async def one(client, model, effort, arm, k):
    p = make_problem(6, 0)
    pr = build(p, arm, k)
    body = {
        "model": model,
        "reasoning": {"effort": effort},
        "store": False,
        "input": [{"role": "developer", "content": pr.developer}, {"role": "user", "content": pr.user}],
    }
    try:
        r = await client.responses.create(**body)
    except Exception as e:  # noqa: BLE001
        print(f"{model:12} {effort:8} {arm}{k}: ERROR {str(e)[:160]}")
        return
    c = check(r.output_text, pr)
    rt = r.usage.output_tokens_details.reasoning_tokens
    print(
        f"{model:12} {effort:8} {arm}{k}: status={r.status} reasoning_tok={rt} out_tok={r.usage.output_tokens} "
        f"compliant={c['compliant']} parsed={c['parsed']} truth={p.answer} diff@{c['diff_at']} | {r.output_text[:60]!r}"
    )
    with out.open("a") as f:  # noqa: ASYNC230
        f.write(
            json.dumps(
                {
                    "model": model,
                    "effort": effort,
                    "arm": arm,
                    "k": k,
                    "body": body,
                    "response": r.model_dump(),
                    "check": c,
                    "truth": p.answer,
                }
            )
            + "\n"
        )


async def main():
    client = AsyncOpenAI(api_key=openai_key(), timeout=600)
    await asyncio.gather(*(one(client, m, e, a, k) for m, e in COMBOS for a, k in CASES))


asyncio.run(main())
