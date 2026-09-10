#!/bin/bash
# AIME-Plus-Plus (post-cutoff) conditions not yet run: reasoning-low (R) and the position ablations (YB / XC / CC).
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
C="--models gpt-6-astra:low --tasks aimepp --depths 1 --n 157 --concurrency 40 --log-every 200"
$R/drivers/run_until_done.sh --tag reasoning_low $C --arms R --ks 0
$R/drivers/run_until_done.sh --tag position_astra $C --arms B YB --ks 0 300 2000
$R/drivers/run_until_done.sh --tag position_astra $C --arms XC --ks 300
$R/drivers/run_until_done.sh --tag position_astra $C --arms CC --ks 300 1000
echo "$(date +%T) ALL DONE"
