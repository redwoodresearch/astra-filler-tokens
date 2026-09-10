#!/usr/bin/env bash
# Relaunch nf.run until it exits cleanly (runs are resumable; native heap crashes have been seen under load).
cd "$(dirname "$0")"
for i in $(seq 1 10); do
  uv run -m nf.run "$@" && exit 0
  echo "$(date +%T) nf.run exited non-zero (attempt $i), relaunching" >&2; sleep 3
done
exit 1
