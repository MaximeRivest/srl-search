#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SRL_FIELDS_GPU=1 CUDA_DEVICE_ORDER=PCI_BUS_ID TRITON_LIBCUDA_PATH=/run/opengl-driver/lib HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
for i in 1 2 3 4 5 6; do ../../../.venv/bin/python fieldpick.py run && exit 0; echo "try $i failed (shared GPU); retrying in 2 min"; sleep 120; done; exit 1
