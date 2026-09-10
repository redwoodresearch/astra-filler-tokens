#!/bin/bash
# HLE (text-only, 2158) + LiveBench answer-format tasks (546): no filler vs counting filler 300 / 1000. Usage: run_bench_hard.sh "<models>" <conc>
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; MODELS="$1"; CONC="$2"
for spec in "hle|2158" "livebench|546"; do
  t=${spec%%|*}; n=${spec##*|}
  $R/drivers/run_until_done.sh --tag bench_hard --models $MODELS --tasks $t --depths 1 --n $n --concurrency $CONC --log-every 500 --arms B CB --ks 0 300 1000
done
echo "$(date +%T) ALL DONE"
