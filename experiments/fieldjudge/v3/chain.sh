#!/usr/bin/env bash
# After the 0.8B finishes (fj3-train): test it, train the 2B (needs ~8.5 GB free on GPU 1), test the 2B.
set -uo pipefail
cd "$(dirname "$0")"
M=/mnt/fast/scholarsreadinglist/fieldjudge/v3/models
until [ -f $M/0.8B/DONE.json ]; do sleep 120; done
while systemctl --user is-active --quiet fj3-train; do sleep 30; done
FJ3_GATE=7000 bash evals.sh "v3-0.8B=$M/0.8B/merged"
FJ3_GATE=8500 FJ3_SIZES=2B bash run3.sh || { echo "2B training stopped"; exit 1; }
FJ3_GATE=9000 bash evals.sh "v3-2B=$M/2B/merged"
echo "ROUND 3 CHAIN DONE"
