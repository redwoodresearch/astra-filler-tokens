#!/bin/bash
# 3-shot prompting test for the four non-Astra models: demonstrations carry the same filler as the query (500 dots or none).
# Cut down to the cells where these models showed any zero-shot filler signal: Gen-Arithmetic 5-7 ops, n-hop 2-3 hops.
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
for M in "or/deepseek/deepseek-v3.2@off" "gpt-5.6-sol:none" "ant/claude-opus-4-5-20251101@off" "ant/claude-opus-5@off"; do
  $R/drivers/run_until_done.sh --tag fewshot3 --models $M --tasks arith --depths 5 6 7 --n 100 --concurrency 30 --log-every 500 --arms B XB --ks 0 500 --shots 3
  $R/drivers/run_until_done.sh --tag fewshot3 --models $M --tasks nhop --depths 2 3 --n 147 --concurrency 30 --log-every 500 --arms B XB --ks 0 500 --shots 3
done
echo "$(date +%T) ALL DONE fewshot3"
