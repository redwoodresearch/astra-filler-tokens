#!/bin/bash
# Framing ablation: XI = dots described as "extra space for you to process the problem" (XB says "carry no information; ignore them").
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; A="gpt-6-astra:low"; S="gpt-5.6-sol:none"
$R/drivers/run_until_done.sh --tag nhop --models $A --tasks nhop --depths 2 3 4 5 6 7 --n 150 --concurrency 40 --log-every 500 --arms XI --ks 256 1024
$R/drivers/run_until_done.sh --tag greenblatt --models $A --tasks arith --depths 15 --n 100 --concurrency 40 --log-every 500 --arms XI --ks 256 1024
$R/drivers/run_until_done.sh --tag greenblatt2 --models $A --tasks aimepp --depths 1 --n 157 --concurrency 40 --log-every 500 --arms XI --ks 256 1024
$R/drivers/run_until_done.sh --tag greenblatt --models $A --tasks aime --depths 1 --n 218 --concurrency 40 --log-every 500 --arms XI --ks 256 1024
$R/drivers/run_until_done.sh --tag nhop --models $S --tasks nhop --depths 4 --n 150 --concurrency 30 --log-every 500 --arms XI --ks 256 1024
$R/drivers/run_until_done.sh --tag greenblatt --models $S --tasks arith --depths 15 --n 100 --concurrency 30 --log-every 500 --arms XI --ks 256 1024
$R/drivers/run_until_done.sh --tag greenblatt2 --models $S --tasks aimepp --depths 1 --n 157 --concurrency 30 --log-every 500 --arms XI --ks 256 1024
echo "$(date +%T) ALL DONE xi"
