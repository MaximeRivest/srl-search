#!/usr/bin/env bash
# eval3.py on GPU 1 for each NAME=MODEL[:legacy] argument, each after the live-map gate.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/maxime/Projects/scholarsreadinglist/search-dev/.venv/bin/python
export SRL_FIELDS_GPU=1 CUDA_DEVICE_ORDER=PCI_BUS_ID TRITON_LIBCUDA_PATH=/run/opengl-driver/lib HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
for spec in "$@"; do
  name=${spec%%=*}; rest=${spec#*=}; model=${rest%%:*}; mode=""; [[ $rest == *:legacy ]] && mode=legacy
  for i in 1 2 3 4; do
    "$PY" ../gate.py "${FJ3_GATE:-9000}"
    "$PY" eval3.py "$name" "$model" $mode && break
    echo "eval $name failed (try $i); retrying in 2 min"; sleep 120
  done
done
