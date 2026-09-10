#!/bin/bash
# Redo the n-hop figures on the rewritten (nested-English) prompts. Astra dots to 8,192 (no 16,384); other models at 4 hops
# (dots to 4,096) and at all hops (no filler + counting 1000). Astra's B / CB300 / CB1000 cells already exist.
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
$R/drivers/run_until_done.sh --tag nhop --models gpt-6-astra:low --tasks nhopnl --depths 2 3 4 5 6 7 --n 150 --concurrency 40 --log-every 500 --arms XB --ks 4 16 64 256 1024 4096 8192
for M in "or/deepseek/deepseek-v3.2@off" "gpt-5.6-sol:none" "ant/claude-opus-4-5-20251101@off" "ant/claude-opus-5@off"; do
  $R/drivers/run_until_done.sh --tag nhop --models $M --tasks nhopnl --depths 4 --n 150 --concurrency 30 --log-every 500 --arms B XB --ks 0 4 16 64 256 1024 4096
  $R/drivers/run_until_done.sh --tag nhop --models $M --tasks nhopnl --depths 2 3 4 5 6 7 --n 150 --concurrency 30 --log-every 500 --arms B CB --ks 0 1000
done
echo "$(date +%T) ALL DONE nhopnl_full"
