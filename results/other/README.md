# `results/other/`: further experiments and ablations

All paths and commands are relative to the repository root.

The few-shot check comes first because it is the most important experiment not in the post; the rest is a catalogue.

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

## Catalogue

Each has figures under `<tag>/figs/` and, where an analysis module exists, the command that regenerates them.

- Synthetic serial-depth suite (permutation composition, table lookup, modular chains, reachability, cellular automata,
  stack, parity): `main`, `tightcap`, `fillerD_*`, `suite`, `suite_deep`, `depth_sweep`; dose curves `dots_dose`,
  `dots_tasks`, `prefix_tasks`, `dots_before` (positional control).
- Many-model sweeps and release-date / time-horizon analyses: `sweep_openai`, `sweep_or`, `sweep_ant`, `filler_trend`,
  `horizon` (`uv run -m nf.analysis.horizon`), `th_real.csv` (`uv run -m nf.analysis.th_real`, rough time horizons with filler).
- Controls on other model families: `dots_plain` (gpt-4o, Llama 3.3, DeepSeek V3), `dots_hybrid`, `dots_rep` (Opus 5,
  Gemini 2.5/3.5/3.8 Flash, Qwen 3.5, Kimi K2.6, Grok 4.20), `fable_dots`, `fable_filler` (Claude Fable 5.1).
- Non-toy benchmarks: `bench_dots`, `bench_filler2` (GSM8K, GPQA, MMLU-Pro).
- Few-shot elicitation: `fewshot_astra`, `fewshot10_astra`, `fewshot10_others`, `ideal_fewshot10` (see above).
- Ablations: `fewshot`, `fewshot3` (few-shot prompting; `uv run -m nf.analysis.fewshot3`), `arith_shape` (chain vs balanced
  arithmetic; `uv run -m nf.analysis.arith_shape`), `framing.csv` (dots described as thinking space; `uv run -m nf.analysis.framing`),
  `filler_methods.csv` (paired tests between filler methods at matched token counts; `uv run -m nf.analysis.filler_methods`),
  `reasoning_low_pilot`, `probe_*`, `smoke_suite`.

