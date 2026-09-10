#!/bin/bash
# Natural-fact n-hop (Greenblatt multi_hop extended to 2-7 hops); user-supplied filler arms only. Usage: run_nhop.sh "<models>" <conc>
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$1"; CONC="$2"
common="--tag nhop --models $MODELS --tasks nhop --depths 2 3 4 5 6 7 --n 150 --concurrency $CONC --log-every 500"
$R/drivers/run_until_done.sh $common --arms B CB --ks 0 10 100 300 1000
$R/drivers/run_until_done.sh $common --arms XB --ks 300
$R/drivers/run_until_done.sh $common --arms RB --ks 1 5
