# `data/`: task datasets and configs

All paths and commands are relative to the repository root.

| file | contents | source |
|---|---|---|
| `nhop_{2..7}.jsonl` | N-hop natural-facts questions, 150 per hop count, with the fact chain | composed from [rgreenblatt/multi_hop](https://github.com/rgreenblatt/multi_hop) fact tables (`scripts/build_nhop.py`; the shipped files are canonical) |
| `nhopnl_{2..7}.jsonl` | the same chains rendered as nested English (`scripts/build_nhop_nl.py`) | |
| `aime.jsonl` | 218 integer-answer problems, AIME/HMMT 2024–26 + BRUMO/CMIMC/SMT 2025 | [MathArena](https://huggingface.co/MathArena) |
| `aimepp_meta.jsonl` | AIME-Plus-Plus ids and tiers only (157 problems, post-cutoff for Astra) | rebuild `data/aimepp.jsonl` with `scripts/build_aimepp.py` from [ulamai/AIME-Plus-Plus](https://huggingface.co/datasets/ulamai/AIME-Plus-Plus) |
| `livebench_meta.jsonl` | LiveBench ids, categories, tasks and subtasks only (618 items) | rebuild `data/livebench.jsonl` (public 2024 releases, embedded CoT/format sentences stripped) with `scripts/build_hle_livebench.py` from [livebench](https://huggingface.co/livebench) |
| `hle_meta.jsonl` | HLE ids, categories, answer types only | HLE is gated: rebuild `data/hle.jsonl` with `scripts/build_hle_livebench.py` and an `HF_TOKEN` |
| `gsm8k.jsonl`, `gpqa.jsonl`, `mmlupro.jsonl`, `lehigh26.jsonl` | further benchmarks used in `results/other` | |
| `us_state_mottos_flowers.json` | tables used by the N-hop grader | rgreenblatt/multi_hop |
| `openrouter_pins.json`, `openrouter_models_catalog.json`, `human_time_ratings.json`, `metr_horizons.csv` | configs for OpenRouter endpoint pinning and the time-horizon analyses | |

The `*_meta.jsonl` files are enough for every analysis of the stored logs; the full question files are needed only to run new evaluations.

Gen-Arithmetic (`arith`, and the `arithchain`/`arithbal` shape variants) and the synthetic serial tasks are generated
on the fly by `src/nf/tasks.py` (a port of Greenblatt's generator: ops `+ - * // %`, integers in [-99, 99]).

