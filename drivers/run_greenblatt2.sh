#!/bin/bash
# Greenblatt arms on the low-contamination sets: AIME-Plus-Plus (157 original problems, Aug 2026) + Lehigh 2026 (11).
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$1"; CONC="$2"
for task_spec in "aimepp|157" "lehigh26|11"; do
  task=${task_spec%%|*}; n=${task_spec##*|}
  common="--tag greenblatt2 --models $MODELS --tasks $task --depths 1 --n $n --concurrency $CONC --log-every 500"
  $R/drivers/run_until_done.sh $common --arms B CB --ks 0 10 30 100 300 1000
  $R/drivers/run_until_done.sh $common --arms XB --ks 300
  $R/drivers/run_until_done.sh $common --arms RB --ks 1 5
done
echo "GREENBLATT2 DONE for $MODELS"
