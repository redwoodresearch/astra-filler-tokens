#!/bin/bash
# Counting filler (CB user / CC model) and whole-question repeats (RB user / C model) on the non-toy benchmarks.
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$1"; CONC="$2"
common="--tag bench_filler2 --tasks gsm8k gpqa mmlupro --depths 1 --n 100 --concurrency $CONC --log-every 500"
$R/drivers/run_until_done.sh --models $MODELS $common --arms B CB --ks 0 10 50 100 500 1000
$R/drivers/run_until_done.sh --models $MODELS $common --arms RB --ks 1 3
$R/drivers/run_until_done.sh --models $MODELS $common --arms CC --ks 10 100
$R/drivers/run_until_done.sh --models $MODELS $common --arms C --ks 1 3
echo "FILLER2 DONE for $MODELS"
