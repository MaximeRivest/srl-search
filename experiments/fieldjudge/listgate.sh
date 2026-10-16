#!/usr/bin/env bash
# Step 9 on GPU 1 (the rewriter on GPU 0 keeps running). Kimi only scores the results.
set -euo pipefail
cd "$(dirname "$0")"
export SRL_FIELDS_GPU=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=/home/maxime/Projects/scholarsreadinglist/search-dev \
       PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=16 \
       SRL_RELEASE=/mnt/fast/scholarsreadinglist/release/releases/20261016T073008Z \
       SRL_VLLM_PYTHON=/nix/store/skvyvs8jj0k7q4vw2137blrnfp3y5cb2-vllm-runtime-0.29.0/bin/python \
       TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
for i in 1 2 3 4 5 6; do /home/maxime/Projects/scholarsreadinglist/search-dev/.venv/bin/python listgate.py && exit 0; echo "try $i failed (the GPU is shared); retrying in 2 min"; sleep 120; done; exit 1
