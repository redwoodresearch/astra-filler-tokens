# Excluded calls behind the post's figures

Every API call is stored as one row in `results/<main|other>/<tag>/<model>.jsonl.gz`, with the request, the raw API
response (including usage), the parsed answer, correctness and a byte-exact compliance check. This note records how many
of those rows the figures in the post leave out, and why. Regenerate the table with `uv run -m nf.exclusions`.

## What "hidden reasoning" means and how it is measured

The main claim rests on the models not reasoning before answering. Reasoning in the visible output is impossible by
construction: the compliance check requires the text before `ANSWER:` to be byte-identical to the expected filler (or
empty), and the output cap (about 1.3 x the expected answer length + 32 tokens) truncates anything longer. Hidden
reasoning is checked from the provider's own accounting:

- **OpenAI (gpt-6-astra, gpt-5.6-*)**: the Responses API returns `usage.output_tokens_details.reasoning_tokens` and lists any
  reasoning item in `response.output`. Both are stored per call. A call with `reasoning_tokens > 0` is excluded from every
  analysis. Astra runs at `reasoning.effort=low` (there is no `none` for it) with a developer message forbidding
  reasoning; the stored counts show this yields 0 reasoning tokens on all but a handful of calls.
- **OpenRouter (deepseek-v3.2 and other open models)**: reasoning is disabled in the request (`reasoning: {enabled: false}`)
  and the returned `reasoning_tokens` is stored; `> 0` is excluded.
- **Anthropic (claude-opus-5, claude-opus-4.5)**: thinking is disabled in the request, so hidden reasoning cannot occur. We
  also store an *estimate* (billed output tokens - o200k count of the visible text - 4) as a sanity check. That estimate
  is a tokenizer artefact on long answers (it reads 11-22 on multi-word names such as "John Bardeen, Leon Cooper, John
  Robert Schrieffer") and is **not** used to exclude rows. An earlier version of the analysis did exclude Anthropic rows
  with an estimate above 10; that dropped 540 Opus 5 N-hop calls that were plain answers, and moved Opus 5's N-hop
  accuracies by at most 0.01 when reinstated.
- Calls in the "reasoning allowed" arm (`R`) are the one place reasoning is permitted, and are analysed separately.

Other exclusions: API `error` rows (never a completed answer); `refusal` rows (Anthropic `stop_reason: refusal`, empty
content; Opus 5 only); `incomplete` rows where the output hit the cap. Truncated rows are excluded from the paired
analyses of the math and N-hop sets (they are long, malformed answers, 79 in total) and are scored as wrong on HLE and
LiveBench, where 165 rows are answers longer than the true answer (the cap there is sized per item from the truth).
Dose figures additionally drop any cell where fewer than half the problems were answered, so a mostly-refused cell does
not appear as a point resting on a handful of problems.

## Counts

| tag | model | rows | error | refusal | truncated | hidden reasoning | excluded | excluded % |
|---|---|---|---|---|---|---|---|---|
| N-hop | claude-opus-4-5-20251101 | 7200 | 0 | 0 | 6 | 0 | 6 | 0.08 |
| N-hop | claude-opus-5 | 7200 | 0 | 0 | 8 | 0 | 8 | 0.11 |
| N-hop | deepseek/deepseek-v3.2 | 7200 | 0 | 0 | 1 | 0 | 1 | 0.01 |
| N-hop | gpt-5.6-sol | 7500 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| N-hop | gpt-6-astra | 25200 | 3 | 0 | 4 | 1 | 8 | 0.03 |
| AIME/HMMT + Gen-Arithmetic | claude-opus-4-5-20251101 | 7062 | 0 | 0 | 3 | 0 | 3 | 0.04 |
| AIME/HMMT + Gen-Arithmetic | claude-opus-5 | 7062 | 0 | 714 | 4 | 0 | 718 | 10.17 |
| AIME/HMMT + Gen-Arithmetic | deepseek/deepseek-v3.2 | 5170 | 0 | 0 | 1 | 0 | 1 | 0.02 |
| AIME/HMMT + Gen-Arithmetic | gpt-5.6-sol | 7262 | 0 | 0 | 2 | 0 | 2 | 0.03 |
| AIME/HMMT + Gen-Arithmetic | gpt-6-astra | 9642 | 0 | 0 | 13 | 0 | 13 | 0.13 |
| AIME-Plus-Plus | claude-opus-4-5-20251101 | 2454 | 0 | 0 | 7 | 0 | 7 | 0.29 |
| AIME-Plus-Plus | claude-opus-5 | 2454 | 0 | 32 | 0 | 0 | 32 | 1.30 |
| AIME-Plus-Plus | deepseek/deepseek-v3.2 | 2355 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| AIME-Plus-Plus | gpt-5.6-sol | 2768 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| AIME-Plus-Plus | gpt-6-astra | 3082 | 0 | 0 | 33 | 0 | 33 | 1.07 |
| HLE + LiveBench | gpt-6-astra | 8328 | 0 | 0 | 165 | 0 | 0 | 0.00 |
| reasoning low | gpt-6-astra | 3351 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| position ablation | gpt-6-astra | 7308 | 0 | 0 | 2 | 1 | 3 | 0.04 |

Total rows: 122598 | excluded: 835 (0.68%)

Of the 835 excluded rows, 746 are Opus 5 refusals, 79 are truncated outputs, and 10 are OpenAI/OpenRouter calls that
reported reasoning tokens (with the API error rows making up the remainder). No figure changes materially if the
truncated rows are instead scored as wrong.

## Dose cells dropped from figures (fewer than half the problems answered)

| tag | model | task | ops | arm | k | answered | of |
|---|---|---|---|---|---|---|---|
| tag | model | task | depth | arm | k | answered | of |
| AIME/HMMT + Gen-Arithmetic | claude-opus-5 | arith | 5 | CB | 1000 | 45 | 100 |
| AIME/HMMT + Gen-Arithmetic | claude-opus-5 | arith | 6 | CB | 30 | 47 | 100 |
| AIME/HMMT + Gen-Arithmetic | claude-opus-5 | arith | 6 | CB | 1000 | 38 | 100 |
| AIME/HMMT + Gen-Arithmetic | claude-opus-5 | arith | 7 | CB | 30 | 41 | 100 |
| AIME/HMMT + Gen-Arithmetic | claude-opus-5 | arith | 7 | CB | 1000 | 43 | 100 |
| AIME/HMMT + Gen-Arithmetic | claude-opus-5 | arith | 10 | CB | 30 | 37 | 100 |
| AIME/HMMT + Gen-Arithmetic | claude-opus-5 | arith | 10 | CB | 1000 | 38 | 100 |
| AIME/HMMT + Gen-Arithmetic | claude-opus-5 | arith | 15 | CB | 30 | 39 | 100 |
| AIME/HMMT + Gen-Arithmetic | claude-opus-5 | arith | 15 | CB | 1000 | 31 | 100 |
| AIME/HMMT + Gen-Arithmetic | claude-opus-5 | arith | 15 | XB | 4096 | 4 | 100 |

All are Opus 5 with thinking off on Gen-Arithmetic, where it refuses most counting-30 / counting-1000 calls at every op
count and 96 of 100 calls with 4,096 dots.
