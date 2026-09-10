#!/bin/bash
# Dot filler at 4^i tokens. Usage: run_dose4.sh astra | others
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; W="$1"
case "$W" in
  astra)
    M="gpt-6-astra:low"; KS="4 16 64 256 1024 4096 16384"
    $R/drivers/run_until_done.sh --tag nhop --models $M --tasks nhop --depths 2 3 4 5 6 7 --n 150 --concurrency 40 --log-every 500 --arms XB --ks $KS
    $R/drivers/run_until_done.sh --tag greenblatt --models $M --tasks arith --depths 15 --n 100 --concurrency 40 --log-every 500 --arms XB --ks $KS
    $R/drivers/run_until_done.sh --tag greenblatt2 --models $M --tasks aimepp --depths 1 --n 157 --concurrency 40 --log-every 500 --arms XB --ks $KS
    ;;
  others)
    KS="4 16 64 256 1024 4096"
    for M in "or/deepseek/deepseek-v3.2@off" "gpt-5.6-sol:none" "ant/claude-opus-4-5-20251101@off" "ant/claude-opus-5@off"; do
      $R/drivers/run_until_done.sh --tag nhop --models $M --tasks nhop --depths 4 --n 150 --concurrency 30 --log-every 500 --arms XB --ks $KS
      $R/drivers/run_until_done.sh --tag greenblatt --models $M --tasks arith --depths 15 --n 100 --concurrency 30 --log-every 500 --arms XB --ks $KS
      $R/drivers/run_until_done.sh --tag greenblatt2 --models $M --tasks aimepp --depths 1 --n 157 --concurrency 30 --log-every 500 --arms XB --ks $KS
    done
    ;;
esac
echo "$(date +%T) ALL DONE dose4 $W"
