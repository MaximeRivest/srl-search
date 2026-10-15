#!/usr/bin/env bash
# Deliberately stop after 2,000 pairs per teacher for a data-quality audit.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=$1
DATA=${FIELDJUDGE_V2:-/mnt/fast/scholarsreadinglist/fieldjudge/v2}
# Restart=on-failure has an 'activating/auto-restart' interval. It is NOT completion.
while true; do
  state=$(systemctl --user show fj-v2-prepare.service --property=ActiveState --value)
  substate=$(systemctl --user show fj-v2-prepare.service --property=SubState --value)
  [ "$state" = active ] && [ "$substate" = exited ] && break  # successful RemainAfterExit unit
  case "$state" in
    active|activating|reloading|deactivating) sleep 30 ;;
    *) break ;;
  esac
done
result=$(systemctl --user show fj-v2-prepare.service --property=Result --value)
[ "$result" = success ] && [ -f "$DATA/pairs-train.jsonl" ] && [ -f "$DATA/pairs-validation.jsonl" ] || {
  echo 'Preparation paused or incomplete; no labeling started. Resume preparation, then this job.'
  exit 78
}
exec /home/maxime/Projects/scholarsreadinglist/sciencemadereadable/.venv/bin/python run_stage.py label.py "$MODEL" train --batch-size 25 --limit-pairs 2000
