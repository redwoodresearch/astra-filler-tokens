#!/bin/bash
# Dot dose-response for hybrid-reasoning models with reasoning OFF, at transition depths (tag dots_hybrid).
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
SPEC=$1
case "$SPEC" in
  ant/claude-opus-5@off) LOOK="3 4"; MOD="4 6";;
  *)                     LOOK="2 3"; MOD="3 4";;
esac
declare -A D=([perm]="2 3" [lookup]="$LOOK" [modchain]="$MOD")
for task in perm lookup modchain; do
  $R/drivers/run_until_done.sh --tag dots_hybrid --models $SPEC --tasks $task --depths ${D[$task]} --n 40 --arms B XB XC --ks 0 1 5 20 70 140 340 700 --concurrency 20 --log-every 500
done
echo "ALL TASKS DONE for $SPEC"
