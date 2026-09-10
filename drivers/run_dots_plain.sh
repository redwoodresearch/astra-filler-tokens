#!/bin/bash
# Dot dose-response on all six clean tasks for a plain-transformer model at its transition depths.
# usage: run_dots_plain.sh <model_spec>   (tag dots_plain; already-done cells are skipped on resume)
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
SPEC=$1
case "$SPEC" in
  gpt-4o)                              XOR="8 12";;
  or/deepseek/deepseek-chat-v3-0324)   XOR="8 12";;
  or/meta-llama/llama-3.3-70b-instruct) XOR="4 6";;
  *) echo "unknown spec"; exit 1;;
esac
declare -A D=([perm]="2 3" [lookup]="2 3" [modchain]="2 3 4" [reach]="2 3" [stack]="4 6" [xor]="$XOR")
for task in perm lookup modchain reach stack xor; do
  $R/drivers/run_until_done.sh --tag dots_plain --models $SPEC --tasks $task --depths ${D[$task]} --n 40 --arms B XB XC --ks 0 1 5 20 70 140 340 700 --concurrency 24 --log-every 500
done
echo "ALL TASKS DONE for $SPEC"
