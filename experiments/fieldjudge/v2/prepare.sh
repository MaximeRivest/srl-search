#!/usr/bin/env bash
# Prepare the full dataset, then STOP. Labels require a passed batching pilot.
# Both environments already exist; this script does not install dependencies.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=16
export SRL_RELEASE=${SRL_RELEASE:-/mnt/fast/scholarsreadinglist/release/releases/20261015T073330Z}
DATA=${FIELDJUDGE_V2:-/mnt/fast/scholarsreadinglist/fieldjudge/v2}
SUB=/home/maxime/Projects/scholarsreadinglist/sciencemadereadable/.venv/bin/python
CPU=/home/maxime/Projects/scholarsreadinglist/search-dev/.venv/bin/python
mkdir -p "$DATA"
exec 9>"$DATA/prepare.lock"
flock -n 9 || exit 1
status() { printf '%s %s\n' "$(date -Is)" "$*" | tee "$DATA/PREPARE_STATUS.txt"; }
trap 'status "STOPPED at line $LINENO; inspect the unit log before resuming"' ERR
# Do not duplicate an already-running topic generation process.
while systemctl --user is-active --quiet fj-v2-topics; do sleep 30; done
status 'topic generation and overlap screening'
"$SUB" topics.py
status 'freeze retrieval benchmark'
"$CPU" collect.py benchmark
status 'freeze title-only benchmark'
"$CPU" collect.py benchmark-titles
status 'group overlapping topic families before choosing train/validation splits'
"$CPU" group_split.py
status 'collect validation families first, then training families'
"$CPU" collect.py families
status 'mine difficult pairs with the existing local models; yields between chunks to live maps'
"$CPU" mine.py
status 'PREPARED: 100000 pairs. Labeling and model training are separate gated stages.'
