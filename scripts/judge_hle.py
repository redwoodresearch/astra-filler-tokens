"""LLM-judge pass over HLE outputs (official HLE judge prompt, judge = gpt-4.1, temperature 0). Writes every judge prompt
and raw output to results/bench_hard/aux/hle_judge.jsonl; resumable. Usage: uv run scripts/judge_hle.py [tag]"""

import asyncio
import json
import sys

from openai import AsyncOpenAI

from nf.analyze import ROOT, load
from nf.bench_grade import last_answer
from nf.keys import openai_key
from nf.tasks import dataset_row

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
"""
OUT = ROOT / "results" / "bench_hard" / "aux" / "hle_judge.jsonl"
JUDGE = "gpt-4.1"


async def one(client, sem, r, f):
    async with sem:
        prompt = JUDGE_PROMPT.format(
            question=dataset_row("hle", int(r.problem_id.split("_i")[1].split("_")[0]))["statement"],
            response=r.output_text or "",
            correct_answer=r.truth,
        )
        resp = await client.chat.completions.create(
            model=JUDGE, temperature=0, messages=[{"role": "user", "content": prompt}]
        )
        txt = resp.choices[0].message.content or ""
        verdict = (
            txt.lower().rsplit("correct:", 1)[-1].strip()[:3].startswith("yes") if "correct:" in txt.lower() else None
        )
        rec = {
            "problem_id": r.problem_id,
            "arm": r.arm,
            "k": int(r.k),
            "model_eff": r.model_eff,
            "truth": r.truth,
            "parsed": last_answer(r.output_text or ""),
            "programmatic": bool(r.correct),
            "judge_model": JUDGE,
            "judge_prompt": prompt,
            "judge_output": txt,
            "judge_correct": verdict,
        }
        f.write(json.dumps(rec) + "\n")
        f.flush()


async def main(tag):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = load(tag)
    df = df[(df.task == "hle") & (df.status == "completed")]
    done = set()
    if OUT.exists():
        done = {(d["problem_id"], d["arm"], d["k"], d["model_eff"]) for d in map(json.loads, OUT.open())}
    todo = [r for r in df.itertuples() if (r.problem_id, r.arm, int(r.k), r.model_eff) not in done]
    print(f"{len(todo)} to judge ({len(done)} done)")
    client = AsyncOpenAI(api_key=openai_key(), timeout=120)
    sem = asyncio.Semaphore(64)
    with OUT.open("a") as f:
        await asyncio.gather(*(one(client, sem, r, f) for r in todo))


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "bench_hard"))
