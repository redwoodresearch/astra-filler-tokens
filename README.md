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

## Layout

```
src/nf/            runner (run.py), prompts, tasks/generators, graders, analysis + figure modules
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
| Astra, N-hop, accuracy vs hops per filler dose | `uv run -m nf.nhop nhop` | `results/main/nhop_astra.png` |
| N-hop across models (4 hops; Astra 4 hops vs others 2 hops; 2 hops; vs hops) | `uv run -m nf.ideal nhop` | `results/main/ideal/nhop_models_*.png` |
| Gen-Arithmetic 15 ops, AIME-Plus-Plus, AIME/HMMT, five models vs filler tokens | `uv run -m nf.ideal nhop` | `results/main/ideal/{arith15,aimepp,aime}_models.png` |
| Kendall tau table | `uv run -m nf.kendall_models` | `results/main/kendall_models.csv` |
| HLE / LiveBench bars and tables | `uv run -m nf.bench_hard`, `uv run -m nf.bench_tables` | `results/main/bench_hard/figs/` |
| No-CoT vs filler vs reasoning at effort low | `uv run -m nf.reasoning_low` | `results/main/reasoning_low/figs/reasoning_low.png` |
| Filler position (after / before / model-emitted) | `uv run -m nf.position` | `results/main/position_astra/position_astra.png` |
| 4x5 grid, all math sets, counting/dots/repeats | `uv run -m nf.greenblatt greenblatt greenblatt2 --compact` | `results/main/greenblatt/figs/greenblatt_replication_4x4.png` |
| Example prompts | `results/main/example_prompts.md`; N-hop rewrite examples in `nhop_rewrite_examples.md` | |

`uv run -m nf.ideal nhopnl` / `uv run -m nf.nhop nhopnl` produce the same N-hop figures from the rewritten (nested-English)
phrasing (`results/main/ideal_nl/`, `nhop_astra_nl.png`); the two phrasings score the same within noise.

Result tags in `results/main/`: `nhop` (N-hop, all models, both phrasings), `greenblatt` (AIME/HMMT + Gen-Arithmetic),
`greenblatt2` (AIME-Plus-Plus + Lehigh 2026), `bench_hard` (HLE + LiveBench; `aux/hle_judge.jsonl.gz` holds the LLM-judge
verdicts), `position_astra` (position ablation), `reasoning_low` (reasoning allowed at effort low).

## `results/other/`: further experiments

Each has figures under `<tag>/figs/` and, where an analysis module exists, the command that regenerates them.

- Synthetic serial-depth suite (permutation composition, table lookup, modular chains, reachability, cellular automata,
  stack, parity): `main`, `tightcap`, `fillerD_*`, `suite`, `suite_deep`, `depth_sweep`; dose curves `dots_dose`,
  `dots_tasks`, `prefix_tasks`, `dots_before` (positional control).
- Many-model sweeps and release-date / time-horizon analyses: `sweep_openai`, `sweep_or`, `sweep_ant`, `filler_trend`,
  `horizon` (`uv run -m nf.horizon`), `th_real.csv` (`uv run -m nf.th_real`, rough time horizons with filler).
- Controls on other model families: `dots_plain` (gpt-4o, Llama 3.3, DeepSeek V3), `dots_hybrid`, `dots_rep` (Opus 5,
  Gemini 2.5/3.5/3.8 Flash, Qwen 3.5, Kimi K2.6, Grok 4.20), `fable_dots`, `fable_filler` (Claude Fable 5.1).
- Non-toy benchmarks: `bench_dots`, `bench_filler2` (GSM8K, GPQA, MMLU-Pro).
- Ablations: `fewshot`, `fewshot3` (few-shot prompting; `uv run -m nf.fewshot3`), `arith_shape` (chain vs balanced
  arithmetic; `uv run -m nf.arith_shape`), `framing.csv` (dots described as thinking space; `uv run -m nf.framing`),
  `filler_methods.csv` (paired tests between filler methods at matched token counts; `uv run -m nf.filler_methods`),
  `reasoning_low_pilot`, `probe_*`, `smoke_suite`.

## Caveats worth knowing before reusing the numbers

- Astra's training cutoff is 2026-04-30. AIME/HMMT (latest Feb 2026), HLE and the public LiveBench items predate it;
  AIME-Plus-Plus (posted 2026-08-26) and everything generated locally do not.
- Exclusions are small and audited (`uv run -m nf.exclusions`; details and the hidden-reasoning check in `docs/EXCLUSIONS.md`): of 122,598 stored calls behind the post's figures,
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
