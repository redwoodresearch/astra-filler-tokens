# Astra is much better at reasoning with filler tokens than previous models

Code, data and per-call logs for the post *Astra is much better at reasoning with filler tokens than previous models*
(Dylan Xu, Sebastian Prasanna, Alek Westover; `post/`). We measure GPT-6 Astra and other models with prompts padded by a
variable number of content-free "filler" tokens (dots, counting, repeated questions) under a no-chain-of-thought protocol,
on N-hop natural-facts questions, generated arithmetic, competition math, HLE and LiveBench.

## Setup

```bash
uv sync
export OPENAI_API_KEY=...        # gpt-6-astra, gpt-5.6-*, gpt-4o
export OPENROUTER_API_KEY=...    # deepseek / llama / qwen / gemini etc. (endpoints pinned in data/openrouter_pins.json)
export ANTHROPIC_API_KEY=...     # claude models (thinking off)
```

Every analysis in this repository runs from the stored logs without any API key.

## Protocol

- **Prompted no-CoT.** A developer message forbids reasoning; the model must reply with one line `ANSWER: <n>`; the output
  cap is set so that hidden reasoning would be truncated. Astra runs at `reasoning.effort=low`; the API-reported
  `reasoning_tokens` is stored for every call and any call with reasoning tokens is excluded. Claude models run with
  thinking off, DeepSeek with reasoning disabled. See `src/nf/prompts.py` (`NO_REASONING`) and `results/main/example_prompts.md`.
- **Filler arms** (`src/nf/prompts.py::build`): `XB` exact number of dots after the statement, `CB` counting
  `Filler: 1 2 … N` (≈2 tokens per number), `RB` k repeats of the question, `B` no filler; model-emitted variants `XC`/`CC`;
  `YB` dots before the statement; `XI` dots framed as "extra space to process the problem"; `R` reasoning allowed.
- **Runner.** `uv run -m nf.run --tag <tag> --models <specs> --tasks <task> --depths … --n … --arms … --ks …`
  writes one JSON row per call (request, full API response, parsed answer, correctness, compliance) to
  `results/<tag>/<model>.jsonl`. `--estimate` prints the call count and token budget without calling any API; the
  `drivers/*.sh` scripts wrap the runs used here. Model specs: `gpt-6-astra:low`, `or/<slug>@off`, `ant/<model>@off`.
  `--shots N` prepends N gold-answer demonstrations in the same arm and dose as the query (see the few-shot section).

## Layout

```
src/nf/            runner (run.py), prompts, tasks/generators, graders, log loader (analyze.py)
src/nf/analysis/   analysis + figure modules, one per experiment (`uv run -m nf.analysis.<name>`)
scripts/           dataset builders (HLE/LiveBench, N-hop composition and rewrite), HLE judge, CoT checks, probes
drivers/           shell drivers for every batch of runs
data/              task datasets and configs (see data/README.md)
results/main/      logs + figures behind the post (body and appendix)
results/other/     further experiments and ablations not in the post
post/              the post (PDF) and its headline figures
```

Logs are stored gzipped (`*.jsonl.gz`); `nf.analyze.load(tag)` reads them. Each row keeps the request, the raw API
response and usage, the parsed answer and a byte-exact compliance check of the pre-answer text.

Each folder documents itself:

- [`results/main/README.md`](results/main/README.md): every figure in the post, the command that regenerates it, and
  the caveats to know before reusing the numbers (training cutoff, exclusions, Opus 5 refusals, HLE scoring).
- [`results/other/README.md`](results/other/README.md): the few-shot (10-shot) elicitation check, then the catalogue of
  further experiments and ablations (serial-depth suite, many-model sweeps, other model families, GSM8K/GPQA/MMLU-Pro).
- [`data/README.md`](data/README.md): where each dataset comes from and how to rebuild the gated or large ones.
- [`docs/EXCLUSIONS.md`](docs/EXCLUSIONS.md): the exclusion audit and hidden-reasoning check.

## Attribution

Filler-token protocol and Gen-Arithmetic / N-hop fact tables: Ryan Greenblatt
([blog post](https://blog.redwoodresearch.org/p/recent-llms-can-use-filler-tokens), [multi_hop](https://github.com/rgreenblatt/multi_hop)).
Counting filler and the no-CoT time-horizon framing: Gould, Ward, Woodruff et al., *Think Fast* (arXiv 2606.07157).
Nested-English N-hop phrasing follows the `realhop_nl` items of [nocot-bench](https://github.com/neelnanda-io/nocot-bench).
