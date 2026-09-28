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

Every analysis below runs from the stored logs without any API key.

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
data/              task datasets (see below) and configs
results/main/      logs + figures behind the post (body and appendix)
results/other/     further experiments and ablations not in the post
post/              the post (PDF) and its headline figures
```

Logs are stored gzipped (`*.jsonl.gz`); `nf.analyze.load(tag)` reads them. Each row keeps the request, the raw API
response and usage, the parsed answer and a byte-exact compliance check of the pre-answer text.

## Data (`data/`)

| file | contents | source |
|---|---|---|
| `nhop_{2..7}.jsonl` | N-hop natural-facts questions, 150 per hop count, with the fact chain | composed from [rgreenblatt/multi_hop](https://github.com/rgreenblatt/multi_hop) fact tables (`scripts/build_nhop.py`; the shipped files are canonical) |
| `nhopnl_{2..7}.jsonl` | the same chains rendered as nested English (`scripts/build_nhop_nl.py`) | |
| `aime.jsonl` | 218 integer-answer problems, AIME/HMMT 2024–26 + BRUMO/CMIMC/SMT 2025 | [MathArena](https://huggingface.co/MathArena) |
| `aimepp_meta.jsonl` | AIME-Plus-Plus ids and tiers only (157 problems, post-cutoff for Astra) | rebuild `data/aimepp.jsonl` with `scripts/build_aimepp.py` from [ulamai/AIME-Plus-Plus](https://huggingface.co/datasets/ulamai/AIME-Plus-Plus) |
| `livebench_meta.jsonl` | LiveBench ids, categories, tasks and subtasks only (618 items) | rebuild `data/livebench.jsonl` (public 2024 releases, embedded CoT/format sentences stripped) with `scripts/build_hle_livebench.py` from [livebench](https://huggingface.co/livebench) |
| `hle_meta.jsonl` | HLE ids, categories, answer types only | HLE is gated: rebuild `data/hle.jsonl` with `scripts/build_hle_livebench.py` and an `HF_TOKEN` |

The `*_meta.jsonl` files are enough for every analysis of the stored logs; the full question files are needed only to run new evaluations.
| `gsm8k.jsonl`, `gpqa.jsonl`, `mmlupro.jsonl`, `lehigh26.jsonl` | further benchmarks used in `results/other` | |
| `us_state_mottos_flowers.json` | tables used by the N-hop grader | rgreenblatt/multi_hop |
| `openrouter_pins.json`, `openrouter_models_catalog.json`, `human_time_ratings.json`, `metr_horizons.csv` | configs for OpenRouter endpoint pinning and the time-horizon analyses | |

Gen-Arithmetic (`arith`, and the `arithchain`/`arithbal` shape variants) and the synthetic serial tasks are generated
on the fly by `src/nf/tasks.py` (a port of Greenblatt's generator: ops `+ - * // %`, integers in [-99, 99]).

## Figures in the post and how to regenerate them

| figure | command | output |
|---|---|---|
| Astra, N-hop, accuracy vs hops per filler dose | `uv run -m nf.analysis.nhop nhop` | `results/main/nhop_astra.png` |
| N-hop across models (4 hops; Astra 4 hops vs others 2 hops; 2 hops; vs hops) | `uv run -m nf.analysis.ideal nhop` | `results/main/ideal/nhop_models_*.png` |
| Gen-Arithmetic 15 ops, AIME-Plus-Plus, AIME/HMMT, five models vs filler tokens | `uv run -m nf.analysis.ideal nhop` | `results/main/ideal/{arith15,aimepp,aime}_models.png` |
| Kendall tau table | `uv run -m nf.analysis.kendall_models` | `results/main/kendall_models.{csv,md}` (`.md` is the appendix layout) |
| HLE / LiveBench bars and tables | `uv run -m nf.analysis.bench_hard`, `uv run -m nf.analysis.bench_tables` | `results/main/bench_hard/figs/` |
| No-CoT vs filler vs reasoning at effort low | `uv run -m nf.analysis.reasoning_low` | `results/main/reasoning_low/figs/reasoning_low.png` |
| Filler position (after / before / model-emitted) | `uv run -m nf.analysis.position` | `results/main/position_astra/position_astra.png` |
| 4x5 grid, all math sets, counting/dots/repeats | `uv run -m nf.analysis.greenblatt greenblatt greenblatt2 --compact` | `results/main/greenblatt/figs/greenblatt_replication_4x4.png` |
| Example prompts | `results/main/example_prompts.md`; N-hop rewrite examples in `nhop_rewrite_examples.md` | |

`uv run -m nf.analysis.ideal nhopnl` / `uv run -m nf.analysis.nhop nhopnl` produce the same N-hop figures from the rewritten (nested-English)
phrasing (`results/main/ideal_nl/`, `nhop_astra_nl.png`); the two phrasings score the same within noise.

Result tags in `results/main/`: `nhop` (N-hop, all models, both phrasings), `greenblatt` (AIME/HMMT + Gen-Arithmetic),
`greenblatt2` (AIME-Plus-Plus + Lehigh 2026), `bench_hard` (HLE + LiveBench; `aux/hle_judge.jsonl.gz` holds the LLM-judge
verdicts), `position_astra` (position ablation), `reasoning_low` (reasoning allowed at effort low).

## Few-shot check: does the effect survive when the format is fully demonstrated?

A natural worry is that part of the zero-shot filler gain is format elicitation: the model is unused to answering
without reasoning, and dots help it settle into the no-CoT format rather than compute anything. To test this, we rerun
the four cross-model comparisons with ten gold-answer demonstrations prepended to every query (`nf.run --shots 10`).
Each demonstration is a held-out problem from the same task, rendered in the same arm with the same dot count as the
query, answered with the gold `ANSWER: <n>` line; the query set is unchanged except that the N-hop set drops the ten
problems used as demos (140 left). Demo sources: arithmetic from seed 1 beyond the query indices, N-hop the last ten
problems of the hop file, AIME/HMMT from AIME-Plus-Plus and vice versa. The Anthropic path caches the demonstration
prefix (`cache_control` on the last demo). 100% compliant, 0 hidden-reasoning calls.

| figure | command | output |
|---|---|---|
| The post's four cross-model panels, 10-shot (N-hop: Astra 4 hops, others 2) | `uv run -m nf.analysis.ideal fewshot10` | `results/other/ideal_fewshot10/{nhop,arith15,aimepp,aime}_models.png` |
| Astra 0- vs 3- vs 10-shot, four tasks | `uv run -m nf.analysis.fewshot_astra` | `results/other/fewshot_astra/figs/fewshot_astra.png` (+ `.csv`) |
| Sol / Opus 4.5 / DeepSeek 0- vs 10-shot, 3x4 grid | `uv run -m nf.analysis.fewshot_astra others` | `results/other/fewshot10_others/figs/fewshot10_others.png` (+ `.csv`) |
| Kendall tau table, 10-shot | `uv run -m nf.analysis.kendall_models fewshot10` | `results/other/ideal_fewshot10/kendall_models_fewshot10.{csv,md}` |

Logs: `results/other/fewshot_astra` (3-shot), `fewshot10_astra` (10-shot, doses to 4,096), `fewshot10_others` (Sol,
Opus 4.5, DeepSeek V3.2 at 10-shot, doses to 1,024; N-hop at 2 and 4 hops). Opus 5 was not run (it refuses few-shot dot
prompts).

Demonstrations raise every model's no-filler baseline and change nothing else. For Astra: 15-op arithmetic 0.50 → 0.75,
4-hop 0.08 → 0.20, AIME/HMMT 0.29 → 0.34 at zero dots, and then the same climb with dots as zero-shot (0.75 → 0.92,
0.20 → 0.51, 0.34 → 0.75 at 4,096; paired 4,096-vs-none wins/losses 17/0, 48/4, 89/1; 3-shot and 10-shot are
indistinguishable). For the others: arithmetic Sol 0.08 → 0.19, Opus 4.5 0.05 → 0.12, Sol 2-hop 0.19 → 0.38, with the
filler response left where it was (Sol small, Opus 4.5 marginal on AIME, DeepSeek none). So the low-dose part of the
zero-shot gain is partly format elicitation, the high-dose part is not.

The post's positive-correlation test on the 10-shot runs (Kendall tau-b, log dose vs per-call correctness, same cells
as the panels; bold p < 0.05; Astra's few-shot N-hop run is 4 hops only):

| Dataset | Astra | 5.6-Sol | Opus 4.5 | Deepseek |
|---|---|---|---|---|
| Gen-Arithmetic 15 ops | **0.15** (p<1e-3) | 0.05 (0.23) | 0.00 (0.95) | 0.01 (0.89) |
| N-hop, 2 hops | — | 0.05 (0.16) | 0.05 (0.16) | −0.03 (0.43) |
| N-hop, 4 hops | **0.20** (p<1e-8) | 0.07 (0.09) | −0.07 (0.07) | −0.02 (0.61) |
| AIME-Plus-Plus, AIME tier | 0.10 (0.15) | 0.06 (0.41) | 0.04 (0.62) | 0.06 (0.42) |
| AIME/HMMT 2024–26 | **0.25** (p<1e-20) | 0.05 (0.10) | 0.03 (0.29) | 0.00 (0.93) |

Astra's taus are smaller than zero-shot (0.28 / 0.29 / 0.22 / 0.29) because the demos raise its baseline and the sweep
has five doses rather than nine, but stay large and significant on the three big sets; AIME-Plus-Plus has 34 problems
and a 0.71 baseline. Sol's two significant zero-shot cells (arithmetic, 2 hops) fall to p ≈ 0.2; no other model
reaches p < 0.05. Caveat: each demo set is a fixed sample, not resampled per query.

## `results/other/`: further experiments

Each has figures under `<tag>/figs/` and, where an analysis module exists, the command that regenerates them.

- Synthetic serial-depth suite (permutation composition, table lookup, modular chains, reachability, cellular automata,
  stack, parity): `main`, `tightcap`, `fillerD_*`, `suite`, `suite_deep`, `depth_sweep`; dose curves `dots_dose`,
  `dots_tasks`, `prefix_tasks`, `dots_before` (positional control).
- Many-model sweeps and release-date / time-horizon analyses: `sweep_openai`, `sweep_or`, `sweep_ant`, `filler_trend`,
  `horizon` (`uv run -m nf.analysis.horizon`), `th_real.csv` (`uv run -m nf.analysis.th_real`, rough time horizons with filler).
- Controls on other model families: `dots_plain` (gpt-4o, Llama 3.3, DeepSeek V3), `dots_hybrid`, `dots_rep` (Opus 5,
  Gemini 2.5/3.5/3.8 Flash, Qwen 3.5, Kimi K2.6, Grok 4.20), `fable_dots`, `fable_filler` (Claude Fable 5.1).
- Non-toy benchmarks: `bench_dots`, `bench_filler2` (GSM8K, GPQA, MMLU-Pro).
- Few-shot elicitation: `fewshot_astra`, `fewshot10_astra`, `fewshot10_others`, `ideal_fewshot10` (see the few-shot
  section above).
- Ablations: `fewshot`, `fewshot3` (few-shot prompting; `uv run -m nf.analysis.fewshot3`), `arith_shape` (chain vs balanced
  arithmetic; `uv run -m nf.analysis.arith_shape`), `framing.csv` (dots described as thinking space; `uv run -m nf.analysis.framing`),
  `filler_methods.csv` (paired tests between filler methods at matched token counts; `uv run -m nf.analysis.filler_methods`),
  `reasoning_low_pilot`, `probe_*`, `smoke_suite`.

## Caveats worth knowing before reusing the numbers

- Astra's training cutoff is 2026-04-30. AIME/HMMT (latest Feb 2026), HLE and the public LiveBench items predate it;
  AIME-Plus-Plus (posted 2026-08-26) and everything generated locally do not.
- Exclusions are small and audited (`uv run -m nf.analysis.exclusions`; details and the hidden-reasoning check in `docs/EXCLUSIONS.md`): of 122,598 stored calls behind the post's figures,
  835 (0.7%) are excluded — 746 Opus 5 refusals, 79 truncated outputs, 10 calls where the OpenAI/OpenRouter API reported
  hidden reasoning tokens. Anthropic calls run with thinking disabled, so none are excluded on the reasoning-token
  estimate (which is a tokenizer artefact on long answers); OpenAI/OpenRouter calls with any reasoning tokens are dropped.
- Claude Opus 5 with thinking off returns `stop_reason: refusal` on some filler cells (e.g. most 4,096-dot calls); refused
  calls are excluded and figures drop dose points where fewer than half the problems were answered.
- HLE is scored with the official judge prompt (gpt-4.1) and, separately, by string match; both are in the tables.
- LiveBench's own "think step by step" / output-format sentences were removed from the statements; the raw first pass with
  them left in is archived under `results/main/bench_hard/aux/`.
- The HLE result rows omit the request/response bodies (which would reproduce the gated question text); every other
  field, including the model's output and the truth, is kept.

## Attribution

Filler-token protocol and Gen-Arithmetic / N-hop fact tables: Ryan Greenblatt
([blog post](https://blog.redwoodresearch.org/p/recent-llms-can-use-filler-tokens), [multi_hop](https://github.com/rgreenblatt/multi_hop)).
Counting filler and the no-CoT time-horizon framing: Gould, Ward, Woodruff et al., *Think Fast* (arXiv 2606.07157).
Nested-English N-hop phrasing follows the `realhop_nl` items of [nocot-bench](https://github.com/neelnanda-io/nocot-bench).
