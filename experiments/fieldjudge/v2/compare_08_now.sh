#!/usr/bin/env bash
# Pause OUR 2B training immediately after a checkpoint, compare, then resume even on evaluation failure.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/maxime/Projects/scholarsreadinglist/search-dev/.venv/bin/python
DATA=${FIELDJUDGE_V2:-/mnt/fast/scholarsreadinglist/fieldjudge/v2}
paused=0
resume() {
 if [[ "$paused" != 1 ]]; then return;fi
 if systemctl --user is-active --quiet fj-v2-models; then return;fi
 if [[ $(systemctl --user show fj-v2-models -p LoadState --value 2>/dev/null || true) == loaded ]]; then
  systemctl --user start fj-v2-models
 else
  systemd-run --user --unit=fj-v2-models -p WorkingDirectory="$PWD" -p Nice=5 -p MemoryMax=40G -p LimitNOFILE=65536 \
   -E PYTHONUNBUFFERED=1 /bin/bash run_models.sh
 fi
}
trap resume EXIT
trap 'exit 143' TERM
if systemctl --user is-active --quiet fj-v2-models; then
 echo 'Waiting for a fresh 2B checkpoint before pausing'
 "$PY" - <<'PY'
import time
from pathlib import Path
p=Path('/mnt/fast/scholarsreadinglist/fieldjudge/v2/models/2B/checkpoint.pt')
done=p.with_name('DONE.json')
for _ in range(180):
 if done.exists() or (p.exists() and time.time()-p.stat().st_mtime<10):break
 time.sleep(5)
else:raise SystemExit('No fresh checkpoint; not interrupting training')
PY
 if [[ ! -f "$DATA/models/2B/DONE.json" ]]; then
  paused=1
  systemctl --user stop fj-v2-models
 fi
fi
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" compare_08.py
