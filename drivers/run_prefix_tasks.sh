#!/bin/bash
# Statement-prefix/repetition dose-response across tasks (arms PB/PC, k = target tokens) at each model's transition depths. usage: run_dots_tasks.sh <model_spec> <tag>
# Depth choices come from the k=0 suite (results/suite): the two depths bracketing the transition per task.
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
SPEC=$1; TAG=$2
case "$SPEC" in
  gpt-6-astra:low) declare -A D=([perm]="6 8 12" [lookup]="4 6 8" [modchain]="8 12" [reach]="4 6 8" [stack]="24 32" [xor]="48 64");;
  gpt-5.6-sol:none) declare -A D=([perm]="3 4 6" [lookup]="3 4" [modchain]="4 6" [reach]="3 4" [stack]="8 12" [xor]="24 32");;
  *) echo "unknown spec"; exit 1;;
esac
for task in perm lookup modchain reach stack xor; do
  $R/drivers/run_until_done.sh --tag $TAG --models $SPEC --tasks $task --depths ${D[$task]} --n 40 --arms B PB PC --ks 0 20 70 140 340 700 --concurrency 32 --log-every 500
done
echo "ALL TASKS DONE for $SPEC"
