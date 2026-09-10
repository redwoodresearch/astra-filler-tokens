#!/bin/bash
# Greenblatt (2025) filler replication on AIME/HMMT 2024-2026 (218 integer-answer problems) and Gen-Arithmetic (300).
# Counting filler 1..N (user), dots at 300 (user), question repeats 1/5 (user). usage: run_greenblatt.sh "<models>" <conc>
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$1"; CONC="$2"
for task_spec in "aime|1|218" "arith|5 6 7|100"; do
  task=${task_spec%%|*}; rest=${task_spec#*|}; depths=${rest%%|*}; n=${rest##*|}
  common="--tag greenblatt --models $MODELS --tasks $task --depths $depths --n $n --concurrency $CONC --log-every 500"
  $R/drivers/run_until_done.sh $common --arms B CB --ks 0 10 30 100 300 1000
  $R/drivers/run_until_done.sh $common --arms XB --ks 300
  $R/drivers/run_until_done.sh $common --arms RB --ks 1 5
done
echo "GREENBLATT DONE for $MODELS"
