#!/bin/bash
# Position ablations on Astra for nhop / 15-op arithmetic / AIME-HMMT: dots before the statement (YB) vs model-emitted
# filler after (empty) reasoning (XC dots, CC counting). User-supplied comparators (XB 300, CB 300/1000) already exist in
# results/greenblatt and results/nhop. Usage: run_position_astra.sh <conc>
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; CONC="$1"
for t in "nhop|2 3 4 5 6 7|150" "arith|15|100" "aime|1|218"; do
  task=${t%%|*}; rest=${t#*|}; depths=${rest%%|*}; n=${rest##*|}
  common="--tag position_astra --models gpt-6-astra:low --tasks $task --depths $depths --n $n --concurrency $CONC --log-every 500"
  $R/drivers/run_until_done.sh $common --arms B YB --ks 0 300 2000
  $R/drivers/run_until_done.sh $common --arms XC --ks 300
  $R/drivers/run_until_done.sh $common --arms CC --ks 300 1000
done
echo "$(date +%T) ALL DONE"
