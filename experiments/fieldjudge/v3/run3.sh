#!/usr/bin/env bash
# Round 3 training: 0.8B then 2B, each from its round-2 judge, on GPU 1, yielding to live maps.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/maxime/Projects/scholarsreadinglist/search-dev/.venv/bin/python
D=/mnt/fast/scholarsreadinglist/fieldjudge
OUT=${FJ3_OUT:-$D/v3/models}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${FJ3_GPU:-1} TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=16
for size in ${FJ3_SIZES:-0.8B 2B}; do
 out="$OUT/$size"
 [ -f "$out/DONE.json" ] && continue
 while true; do
  [ "${FJ3_GPU:-1}" = 0 ] || "$PY" ../gate.py "${FJ3_GATE:-9000}"     # GPU 0: no live maps there
  code=0
  "$PY" train3.py --base "$D/v2/models/$size/merged" --out "$out" || code=$?
  case "$code" in
   0) break ;;
   3) echo 'Yielded GPU to a live map; will resume'; sleep 30 ;;
   *) echo "Training stopped (exit $code); inspect"; exit 78 ;;
  esac
 done
done
echo 'Round 3 trained; nothing deployed.'
