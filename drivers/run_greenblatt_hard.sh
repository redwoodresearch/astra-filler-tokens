#!/bin/bash
# Harder Gen-Arithmetic (10 and 15 operations) with the Greenblatt filler arms. usage: run_greenblatt_hard.sh "<models>" <conc>
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$1"; CONC="$2"
common="--tag greenblatt --models $MODELS --tasks arith --depths 10 15 --n 100 --concurrency $CONC --log-every 500"
$R/drivers/run_until_done.sh $common --arms B CB --ks 0 10 30 100 300 1000
$R/drivers/run_until_done.sh $common --arms XB --ks 300
$R/drivers/run_until_done.sh $common --arms RB --ks 1 5
echo "GREENBLATT-HARD DONE for $MODELS"
