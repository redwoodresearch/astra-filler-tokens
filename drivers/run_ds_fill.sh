#!/bin/bash
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; DS="or/deepseek/deepseek-v3.2@off"
for spec in "greenblatt|aime|1|218" "greenblatt2|aimepp|1|157" "greenblatt|arith|15|100"; do
  IFS='|' read tag task d n <<< "$spec"
  C="--tag $tag --models $DS --tasks $task --depths $d --n $n --concurrency 30 --log-every 500"
  $R/drivers/run_until_done.sh $C --arms B CB --ks 0 10 30 100 300 1000
  $R/drivers/run_until_done.sh $C --arms XB --ks 300
  $R/drivers/run_until_done.sh $C --arms RB --ks 1 5
done
echo "$(date +%T) ALL DONE dsfill"
