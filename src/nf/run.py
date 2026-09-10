"""Run the filler-token experiment grid and stream every call to JSONL.

usage: uv run -m nf.run --tag pilot --models gpt-6-astra:low gpt-5.6-sol:none \
           --depths 4 8 12 --n 20 --arms B C --ks 0 1 2 4 [--cap-mult 1.3 --cap-add 32 | --no-cap]

Model spec is <model_id>:<reasoning_effort>; effort "none" is sent as reasoning.effort="none".
Every response is stored verbatim (request body + full response model_dump) so nothing is lost."""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import anthropic
import tiktoken
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from .bench_grade import grade_hle, grade_livebench, last_answer
from .keys import openai_key, secret_field
from .nhop_grade import check_answer as nhop_check
from .openrouter import PINS, parse_spec
from .prompts import _norm_answer, build, check
from .tasks import dataset_row, make_problem, make_problems

ENC = tiktoken.get_encoding("o200k_base")
ROOT = Path(__file__).resolve().parents[2]


def n_tokens(s: str) -> int:
    return len(ENC.encode(s))


class FakeClient:
    """Dry-run stand-in: answers randomly, always compliant; sometimes 'reasons'."""

    class responses:
        @staticmethod
        async def create(**body):
            await asyncio.sleep(0.01)
            user = body["input"][-1]["content"]
            k, s = body["_k"], body["_stmt"]
            out = (
                "\n\n".join([s] * k) + "\n" if k and body["_arm"] == "C" else ""
            ) + f"ANSWER: {random.choice(['1', 'yes', '0'])}"
            return type(
                "R",
                (),
                {
                    "output_text": out,
                    "status": "completed",
                    "incomplete_details": None,
                    "id": "fake",
                    "model_dump": lambda self: {"fake": True, "output_text": out},
                    "usage": type(
                        "U",
                        (),
                        {
                            "input_tokens": n_tokens(user),
                            "output_tokens": n_tokens(out),
                            "output_tokens_details": type("D", (), {"reasoning_tokens": 0})(),
                        },
                    )(),
                },
            )()


async def call(client, model, effort, prompt, cap, extra, retries=6):
    body = {
        "model": model,
        "input": [{"role": "developer", "content": prompt.developer}]
        + [
            m
            for su, sa in prompt.shots
            for m in ({"role": "user", "content": su}, {"role": "assistant", "content": sa})
        ]
        + [{"role": "user", "content": prompt.user}],
        "store": False,
    }
    if effort:
        body["reasoning"] = {"effort": effort}
    if cap is not None:
        body["max_output_tokens"] = cap
    for attempt in range(retries):
        t0 = time.time()
        try:
            resp = await client.responses.create(**body, **extra)
            return body, resp, time.time() - t0, None
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            await asyncio.sleep(min(60, 2**attempt + random.random()))
            err = repr(e)[:500]
        except APIStatusError as e:
            if e.status_code >= 500 and attempt < retries - 1:
                await asyncio.sleep(2**attempt)
                err = repr(e)[:500]
                continue
            return body, None, time.time() - t0, repr(e)[:1000]
    return body, None, None, err


async def call_openrouter(client, model, effort, prompt, cap, retries=6):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": prompt.developer}]
        + [
            m
            for su, sa in prompt.shots
            for m in ({"role": "user", "content": su}, {"role": "assistant", "content": sa})
        ]
        + [{"role": "user", "content": prompt.user}],
        "extra_body": {
            "provider": PINS.get(model, {"data_collection": "deny", "require_parameters": False}),
            "usage": {"include": True},
        },
    }
    if effort == "off":
        body["extra_body"]["reasoning"] = {"enabled": False}
    elif effort:
        body["extra_body"]["reasoning"] = {"effort": effort}
    if cap is not None:
        body["max_tokens"] = cap
    err = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            resp = await client.chat.completions.create(**body)
            if not resp.choices:  # OpenRouter returns 200 with an error payload on some provider failures
                raise APIConnectionError(request=None, message=f"empty choices: {getattr(resp, 'error', None)}")
            return body, resp, time.time() - t0, None
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            await asyncio.sleep(min(60, 2**attempt + random.random()))
            err = repr(e)[:500]
        except APIStatusError as e:
            if e.status_code >= 500 and attempt < retries - 1:
                await asyncio.sleep(2**attempt)
                err = repr(e)[:500]
                continue
            return body, None, time.time() - t0, repr(e)[:1000]
    return body, None, None, err


async def call_anthropic(client, model, effort, prompt, cap, retries=6):
    """Messages API. effort 'off' -> thinking disabled. Any other effort -> thinking left on (Fable: always on;
    Opus 5 / Sonnet 5: adaptive) with output_config.effort, so the cap is what limits hidden thinking."""
    body = {
        "model": model,
        "max_tokens": cap if cap is not None else 4096,
        "system": prompt.developer,
        "messages": [
            m
            for su, sa in prompt.shots
            for m in ({"role": "user", "content": su}, {"role": "assistant", "content": sa})
        ]
        + [{"role": "user", "content": prompt.user}],
    }
    if effort == "off":
        body["thinking"] = {"type": "disabled"}
    elif effort:
        body["output_config"] = {"effort": effort}
    err = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            resp = await client.messages.create(**body)
            return body, resp, time.time() - t0, None
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            await asyncio.sleep(min(60, 2**attempt + random.random()))
            err = repr(e)[:500]
        except anthropic.APIStatusError as e:
            if e.status_code >= 500 and attempt < retries - 1:
                await asyncio.sleep(2**attempt)
                err = repr(e)[:500]
                continue
            return body, None, time.time() - t0, repr(e)[:1000]
    return body, None, None, err


def _text(resp) -> str:
    if hasattr(resp, "output_text"):
        return resp.output_text or ""
    if hasattr(resp, "stop_reason"):  # anthropic Message
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return resp.choices[0].message.content or ""


def with_shots(prompt, p, arm, k, n):
    """Few-shot demonstrations: held-out problems of the same kind, built with the same arm/filler, answered correctly.
    arith / arithchain / arithbal: same generator at seed 1 (disjoint from the seed-0 evaluation set).
    aimepp: public AIME/HMMT problems (a different dataset from the post-cutoff evaluation set)."""
    from dataclasses import replace

    shots = []
    for j in range(n):
        if p.task == "aimepp":  # demonstrations from the public AIME/HMMT set
            q = make_problem(1, 200 + j, task="aime")
        elif p.task == "aime":  # demonstrations from AIME-Plus-Plus
            q = make_problem(1, j, task="aimepp")
        elif p.task == "nhop":  # last 3 problems of the same hop count; evaluate with --n 147 so they are held out
            q = make_problem(p.depth, 147 + j, task="nhop")
        else:
            q = make_problem(p.depth, 5000 + j, seed=1, task=p.task)
        sp = build(q, arm, k)
        shots.append((sp.user, (sp.expected_prefix + "\n" if sp.expected_prefix else "") + f"ANSWER: {q.answer}"))
    return replace(prompt, shots=tuple(shots))


def _answer_stub(p) -> str:
    """Stand-in answer for sizing the output cap: the true answer when it is longer than a word (HLE lists/expressions),
    else 'yes'. Using the truth only sets the cap; the model never sees it."""
    return str(p.answer) if len(str(p.answer).split()) > 1 or len(str(p.answer)) > 12 else "yes"


def _correct(problem, chk, text) -> bool:
    idx = int(problem.problem_id.split("_i")[1].split("_")[0])
    if problem.task in ("nhop", "nhopnl"):
        return nhop_check(chk["parsed"] or "", problem.answer)
    if problem.task == "hle":
        return grade_hle(last_answer(text), problem.answer, dataset_row("hle", idx)["answer_type"])
    if problem.task == "livebench":
        row = dataset_row("livebench", idx)
        return grade_livebench(last_answer(text), problem.answer, row["task"], row.get("subtask"))
    return chk["parsed"] == _norm_answer(str(problem.answer))


def _reasoning_tokens(resp):
    u = resp.usage
    if hasattr(resp, "stop_reason"):  # anthropic: thinking tokens are billed as output but not itemised -> estimate
        return max(0, u.output_tokens - n_tokens(_text(resp)) - 2)
    for attr in ("output_tokens_details", "completion_tokens_details"):
        d = getattr(u, attr, None)
        if d is not None and getattr(d, "reasoning_tokens", None) is not None:
            return d.reasoning_tokens
    return None


def record(model, effort, problem, prompt, body, resp, latency, err, cap):
    rec = {
        "model": model,
        "effort": effort,
        "problem_id": problem.problem_id,
        "depth": problem.depth,
        "arm": prompt.arm,
        "k": prompt.k,
        "truth": problem.answer,
        "task": problem.task,
        "chance": problem.chance,
        "cap": cap,
        "filler_tokens": prompt.filler_tokens,
        "latency_s": latency,
        "request": body,
        "error": err,
    }
    if resp is None:
        rec.update(status="error", output_text=None, parsed=None, correct=None, compliant=False, reasoning_tokens=None)
        return rec
    usage = resp.usage
    text = _text(resp)
    chk = check(text, prompt)
    is_chat = hasattr(resp, "choices")
    is_ant = hasattr(resp, "stop_reason")
    finish = resp.choices[0].finish_reason if is_chat else (resp.stop_reason if is_ant else None)
    if is_ant:
        status = {"max_tokens": "incomplete", "refusal": "refusal"}.get(resp.stop_reason, "completed")
    elif is_chat:
        status = "incomplete" if finish == "length" else "completed"
    else:
        status = resp.status
    rec.update(
        status=status,
        incomplete_reason=finish
        if (is_chat or is_ant)
        else getattr(getattr(resp, "incomplete_details", None), "reason", None),
        output_text=text,
        served_provider="anthropic" if is_ant else getattr(resp, "provider", None),
        generation_id=getattr(resp, "id", None),
        input_tokens=getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None),
        reasoning_tokens=_reasoning_tokens(resp),
        expected_output_tokens=n_tokens(prompt.expected_prefix + "\nANSWER: yes"),
        **chk,
        correct=_correct(problem, chk, text),
        response=resp.model_dump() if hasattr(resp, "model_dump") else None,
    )
    return rec


async def main(a):
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(a), indent=1))
    problems = make_problems(a.depths, a.n, a.seed, tuple(a.tasks))
    specs = [parse_spec(s) for s in a.models]
    client = (
        FakeClient()
        if a.dry_run
        else (
            AsyncOpenAI(api_key=openai_key(a.key_env), timeout=600, max_retries=0)
            if any(s[0] == "openai" for s in specs)
            else None
        )
    )
    ant_client = None
    if any(s[0] == "anthropic" for s in specs):
        ant_client = anthropic.AsyncAnthropic(api_key=secret_field("ANTHROPIC_API_KEY"), timeout=600, max_retries=0)
    or_client = None
    if any(s[0] == "openrouter" for s in specs):
        or_client = AsyncOpenAI(
            api_key=secret_field("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            timeout=600,
            max_retries=0,
            default_headers={"HTTP-Referer": "https://github.com/neuralese-filler", "X-Title": "neuralese-filler"},
        )
    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    log = (out_dir / "progress.log").open("a")

    def plog(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    dose_run = any(x in ("XB", "XC", "XI", "YB", "PB", "PC", "CB", "CC", "RB") for x in a.arms)
    jobs = []
    for spec in a.models:
        provider, model, effort = parse_spec(spec)
        path = out_dir / f"{model.replace('/', '_')}__{effort}.jsonl"
        done = set()
        if path.exists():
            for line in path.open():
                r = json.loads(line)
                if r.get("status") != "error":
                    done.add((r["problem_id"], r["arm"], r["k"]))
        for p in problems:
            for arm in a.arms:
                for k in a.ks:
                    if arm not in ("B", "R") and k == 0:
                        continue  # k=0 is the shared baseline; only arm B (and the reasoning-allowed arm R) run it
                    if arm == "R" and k > 0:
                        continue  # R has no filler
                    if arm == "B" and k > 0 and dose_run:
                        continue  # in dose runs k is a token/dot count, not a copy count: B is baseline-only
                    if (p.problem_id, arm, k) not in done:
                        jobs.append((provider, model, effort, path, p, arm, k))
    plog(f"{len(jobs)} calls to make ({len(problems)} problems x arms {a.arms} x ks {a.ks} x {len(a.models)} models)")
    # Pre-flight budget: input tokens and expected compliant output tokens over all jobs (o200k estimate).
    est_in = est_out = 0
    per_model = {}
    for _prov, model, effort, _path, p, arm, k in jobs:
        pr = build(p, arm, k)
        i, o = n_tokens(pr.developer) + n_tokens(pr.user) + 20, n_tokens(pr.expected_prefix + "\nANSWER: 1") + 3
        est_in += i
        est_out += o
        m = per_model.setdefault(f"{model}:{effort}", [0, 0, 0])
        m[0] += 1
        m[1] += i
        m[2] += o
    plog(
        f"ESTIMATE: input {est_in / 1e6:.2f}M tokens, output {est_out / 1e6:.2f}M tokens; max single prompt "
        f"{max((n_tokens(build(p, arm, k).user) for *_r, p, arm, k in jobs), default=0)} tokens"
    )
    for m, (n, i, o) in per_model.items():
        plog(f"  {m}: {n} calls, {i / 1e6:.2f}M in, {o / 1e6:.2f}M out")
    if a.estimate:
        plog("estimate-only run; exiting before any API call")
        return
    stats = {"n": 0, "correct": 0, "compliant": 0, "err": 0, "rt": 0}

    async def one(provider, model, effort, path, p, arm, k):
        prompt = build(p, arm, k)
        if a.shots:
            prompt = with_shots(prompt, p, arm, k, a.shots)
        cap = (
            None
            if a.no_cap or arm == "R"
            else max(
                16, int(n_tokens(prompt.expected_prefix + "\nANSWER: " + _answer_stub(p)) * a.cap_mult + a.cap_add)
            )
        )
        extra = {"_k": k, "_arm": arm, "_stmt": p.statement} if a.dry_run else {}
        async with sem:
            if provider == "openrouter":
                body, resp, lat, err = await call_openrouter(or_client, model, effort, prompt, cap)
            elif provider == "anthropic":
                body, resp, lat, err = await call_anthropic(ant_client, model, effort, prompt, cap)
            else:
                body, resp, lat, err = await call(client, model, effort, prompt, cap, extra)
        rec = record(model, effort, p, prompt, body, resp, lat, err, cap)
        rec["provider"] = provider
        async with lock:
            with path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            stats["n"] += 1
            stats["correct"] += bool(rec["correct"])
            stats["compliant"] += bool(rec["compliant"])
            stats["err"] += rec["status"] == "error"
            stats["rt"] += rec.get("reasoning_tokens") or 0
            if stats["n"] % a.log_every == 0 or stats["n"] == len(jobs):
                plog(
                    f"{stats['n']}/{len(jobs)} acc={stats['correct'] / stats['n']:.2f} "
                    f"compliant={stats['compliant'] / stats['n']:.2f} err={stats['err']} "
                    f"mean_reasoning_tok={stats['rt'] / stats['n']:.1f}"
                )
            if rec["status"] == "error" and stats["err"] <= 5:
                plog(f"ERROR {model} {p.problem_id} {arm}{k}: {err}")

    random.shuffle(jobs)  # interleave models/arms so partial results are balanced
    await asyncio.gather(*(one(*j) for j in jobs))
    plog("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--depths", nargs="+", type=int, default=[4, 8, 12])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", nargs="+", default=["perm"])
    ap.add_argument(
        "--shots",
        type=int,
        default=0,
        help="few-shot demonstrations (held-out problems, same arm/filler) before the query",
    )
    ap.add_argument("--arms", nargs="+", default=["B", "C"])
    ap.add_argument("--ks", nargs="+", type=int, default=[0, 1, 2, 4])
    ap.add_argument("--cap-mult", type=float, default=1.3)
    ap.add_argument("--cap-add", type=int, default=32)
    ap.add_argument("--no-cap", action="store_true")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--estimate", action="store_true", help="print call/token budget and exit without calling any API")
    ap.add_argument("--key-env", default=None, help="env var / .envrc name holding the key")
    asyncio.run(main(ap.parse_args()))
