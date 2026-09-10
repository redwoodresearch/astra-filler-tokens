#!/bin/bash
# Astra at reasoning effort low with reasoning allowed (arm R: no no-reasoning instruction, no filler, no cap) on the
# datasets used in the filler study. Usage: run_reasoning_low.sh <conc>
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; CONC="$1"
for spec in "arith|15|100" "arith|10|100" "aime|1|218" "livebench|1|546" "hle|1|2158"; do
  t=${spec%%|*}; rest=${spec#*|}; d=${rest%%|*}; n=${rest##*|}
  $R/drivers/run_until_done.sh --tag reasoning_low --models gpt-6-astra:low --tasks $t --depths $d --n $n --concurrency $CONC --log-every 200 --arms R --ks 0
done
echo "$(date +%T) ALL DONE"
