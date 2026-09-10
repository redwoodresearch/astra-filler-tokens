#!/bin/bash
# Second hybrid batch (tag dots_hybrid, resume-safe): new models + extra depths/reach for Opus 5 and Gemini 2.5 Flash.
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
SPEC=$1; ARMS=$2
case "$SPEC" in
  or/google/gemini-3.5-flash@low) declare -A D=([perm]="2 3" [lookup]="3 4" [modchain]="3 4" [reach]="2 3");;
  or/google/gemini-3.8-flash@low) declare -A D=([perm]="3 4" [lookup]="3 4" [modchain]="3 4" [reach]="2 3");;
  or/x-ai/grok-4.20@off)          declare -A D=([perm]="2 3" [lookup]="2 3" [modchain]="2 3" [reach]="2 3");;
  or/google/gemini-2.5-flash@off) declare -A D=([perm]="4" [lookup]="4" [modchain]="6" [reach]="2 3");;
  ant/claude-opus-5@off)          declare -A D=([perm]="4" [lookup]="6" [modchain]="8" [reach]="3 4");;
  *) echo "unknown spec"; exit 1;;
esac
for task in perm lookup modchain reach; do
  $R/drivers/run_until_done.sh --tag dots_hybrid --models $SPEC --tasks $task --depths ${D[$task]} --n 40 --arms $ARMS --ks 0 1 5 20 70 140 340 700 --concurrency 20 --log-every 500
done
echo "HYBRID2 DONE for $SPEC"
