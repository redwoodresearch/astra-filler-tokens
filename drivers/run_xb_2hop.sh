#!/bin/bash
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
for M in "or/deepseek/deepseek-v3.2@off" "gpt-5.6-sol:none" "ant/claude-opus-4-5-20251101@off" "ant/claude-opus-5@off"; do
  $R/drivers/run_until_done.sh --tag nhop --models $M --tasks nhop --depths 2 --n 150 --concurrency 30 --log-every 500 --arms XB --ks 4 16 64 256 1024 4096
done
echo "$(date +%T) ALL DONE xb2hop"
