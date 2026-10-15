#!/usr/bin/env bash
# Deliberately stop after 2,000 pairs per teacher for a data-quality audit.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=$1
while systemctl --user is-active --quiet fj-v2-prepare; do sleep 60; done
DATA=${FIELDJUDGE_V2:-/mnt/fast/scholarsreadinglist/fieldjudge/v2}
[ -f "$DATA/pairs-train.jsonl" ] || { echo 'Preparation incomplete; no labeling started'; exit 1; }
exec /home/maxime/Projects/scholarsreadinglist/sciencemadereadable/.venv/bin/python label.py "$MODEL" train --batch-size 25 --limit-pairs 2000
