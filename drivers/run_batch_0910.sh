#!/bin/bash
# 2026-09-10 batch. Usage: run_batch_0910.sh <which>   which in {fewshot, nhop, shape}
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; W="$1"
DS="or/deepseek/deepseek-v3.2@off"
case "$W" in
  fewshot)
    # DeepSeek V3.2 (reasoning off), zero-shot into the existing Greenblatt tags, then 5-shot into tag fewshot
    $R/drivers/run_until_done.sh --tag greenblatt2 --models $DS --tasks aimepp --depths 1 --n 157 --concurrency 30 --log-every 500 --arms B CB --ks 0 300 1000
    $R/drivers/run_until_done.sh --tag greenblatt --models $DS --tasks arith --depths 5 6 7 10 15 --n 100 --concurrency 30 --log-every 500 --arms B CB --ks 0 300 1000
    $R/drivers/run_until_done.sh --tag fewshot --models $DS --tasks aimepp --depths 1 --n 157 --concurrency 30 --log-every 500 --arms B CB --ks 0 300 1000 --shots 5
    $R/drivers/run_until_done.sh --tag fewshot --models $DS --tasks arith --depths 5 6 7 10 15 --n 100 --concurrency 30 --log-every 500 --arms B CB --ks 0 300 1000 --shots 5
    ;;
  nhop)
    for M in $DS "gpt-5.6-sol:none" "ant/claude-opus-4-5-20251101@off" "ant/claude-opus-5@off"; do
      $R/drivers/run_until_done.sh --tag nhop --models $M --tasks nhop --depths 2 3 4 5 6 7 --n 150 --concurrency 30 --log-every 500 --arms B CB --ks 0 300 1000
    done
    ;;
  shape)
    $R/drivers/run_until_done.sh --tag arith_shape --models gpt-6-astra:low --tasks arithchain arithbal --depths 7 10 15 --n 100 --concurrency 40 --log-every 500 --arms B CB --ks 0 300 1000
    ;;
esac
echo "$(date +%T) ALL DONE $W"
