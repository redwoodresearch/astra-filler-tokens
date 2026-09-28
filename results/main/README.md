# `results/main/`: logs and figures behind the post

All paths and commands are relative to the repository root.

Each `<tag>/` holds one gzipped log per model (read with `nf.analyze.load(tag)`) and the figures drawn from it; the
tags are listed at the end of the figures section.

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

