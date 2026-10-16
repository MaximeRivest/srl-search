#!/usr/bin/env bash
# Independent overnight training: initial snapshot first, final labels next epoch.
# Uses the same two snapshots for both sizes, preserving comparison fairness.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/maxime/Projects/scholarsreadinglist/search-dev/.venv/bin/python
DATA=${FIELDJUDGE_V2:-/mnt/fast/scholarsreadinglist/fieldjudge/v2}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=16
[ -f "$DATA/training/initial-manifest.json" ] || { echo 'Initial data audit not ready';exit 78; }
for size in 0.8B 2B; do
 out="$DATA/models/$size"
 [ -f "$out/DONE.json" ] && continue
 while true; do
  "$PY" ../gate.py 10000
  code=0
  "$PY" train_incremental.py --base "Qwen/Qwen3.5-$size" --out "$out" || code=$?
  case "$code" in
   0) break ;;
   3) echo 'Yielded GPU to a live map; will resume'; sleep 30 ;;
   4) echo 'First pass complete; waiting for final labels without holding GPU'
      while [ ! -f "$DATA/training/final-manifest.json" ]; do sleep 30; done ;;
   *) echo "Training stopped (exit $code); inspect rather than silently skipping it";exit 78 ;;
  esac
 done
done
echo 'Both models trained; no deployment performed.'
