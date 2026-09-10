#!/bin/bash
# retry loop around the (resumable) HLE judge; native heap aborts have been seen under load
cd "$(dirname "$0")/.."
for i in 1 2 3 4 5 6; do uv run scripts/judge_hle.py bench_hard && exit 0; echo "$(date +%T) judge exited non-zero (attempt $i)"; sleep 3; done
exit 1
