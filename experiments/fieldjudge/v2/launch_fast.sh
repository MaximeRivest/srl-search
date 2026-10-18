#!/usr/bin/env bash
# Start/resume CPU collection and both streaming teachers as independent jobs.
set -euo pipefail
cd "$(dirname "$0")"
SUB=/home/maxime/Projects/scholarsreadinglist/sciencemadereadable/.venv/bin/python
CPU=/home/maxime/Projects/scholarsreadinglist/search-dev/.venv/bin/python
DATA=${FIELDJUDGE_V2:-/mnt/fast/scholarsreadinglist/fieldjudge/v2}
# Teacher roles A and B (names used on disk) come from the private settings (../teachers.py).
read -r A B < <("$SUB" -c 'import sys; sys.path.insert(0, ".."); import teachers; print(*teachers.ROLES)')
[[ -n ${B:-} ]] || { echo 'teacher roles unreadable (../teachers.py)'; exit 78; }
# Supersedes the staged/order-check run; never let both annotation protocols run accidentally.
# (Units used to be named after the roles: those names are stopped too.)
systemctl --user stop fj-v2-prepare fj-v2-first-a fj-v2-first-b "fj-v2-first-$A" "fj-v2-first-$B" 2>/dev/null || true
if [[ ${1:-} == --restart ]]; then
  systemctl --user stop fj-fast-collect fj-fast-a fj-fast-b "fj-fast-$A" "fj-fast-$B" 2>/dev/null || true
fi
"$SUB" fast.py seed
"$SUB" topics.py --available-only
start() {
 local unit=$1 py=$2 script=$3; shift 3
 local state
 state=$(systemctl --user show "$unit" --property=ActiveState --value 2>/dev/null || true)
 if [[ "$state" == active || "$state" == activating ]]; then echo "$unit already running or waiting to retry";return;fi
 systemctl --user reset-failed "$unit" 2>/dev/null || true
 systemd-run --user --unit="$unit" -p WorkingDirectory="$PWD" -p Nice=10 -p CPUWeight=30 \
  -p MemoryMax=60G -p LimitNOFILE=262144 -p Restart=on-failure -p RestartSec=300 \
  -p RestartPreventExitStatus=78 -p RemainAfterExit=yes -p StartLimitIntervalSec=3600 -p StartLimitBurst=8 \
  -E FIELDJUDGE_V2="$DATA" -E PYTHONUNBUFFERED=1 -E HF_HUB_OFFLINE=1 -E TOKENIZERS_PARALLELISM=false \
  -E OPENBLAS_NUM_THREADS=16 -E SRL_RELEASE=/mnt/fast/scholarsreadinglist/release/releases/20261015T073330Z \
  "$py" run_stage.py "$script" "$@"
}
start fj-fast-a "$SUB" fast.py run --model "$A"
start fj-fast-b "$SUB" fast.py run --model "$B"
start fj-fast-collect "$CPU" fast_collect.py
