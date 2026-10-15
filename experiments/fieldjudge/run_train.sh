#!/usr/bin/env bash
# Train (train.py) on GPU 1, stepping aside whenever a whole-field map needs it (exit code 3), then resuming.
#   ./run_train.sh OUT_DIR [train.py options]        e.g. ./run_train.sh /mnt/fast/scholarsreadinglist/fieldjudge/m08
set -u
cd "$(dirname "$0")"
PY=../../.venv/bin/python
OUT=$1; shift
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       TRITON_LIBCUDA_PATH=/run/opengl-driver/lib HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
RESUME=""
[ -f "$OUT/checkpoint.pt" ] && RESUME=--resume
while true; do
  $PY gate.py 10000
  $PY train.py --out "$OUT" $RESUME "$@"
  code=$?
  [ $code -eq 3 ] || exit $code
  RESUME=--resume
  sleep 30
done
