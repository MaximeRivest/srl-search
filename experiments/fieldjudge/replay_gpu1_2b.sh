#!/usr/bin/env bash
# Free (no Kimi, no Jev inside the maps) replay of the round-2 2B on GPU 1, then the reading-list check.
# Same topics, same data release and same output folder as the round-2 0.8B run (fields-r4), so the two are matched.
# GPU 1 instead of GPU 0: the public rewriter keeps running; GPU 1 is the card production maps use anyway.
# Kimi is used only afterwards, to SCORE reading lists, never inside the maps.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/maxime/Projects/scholarsreadinglist/search-dev/.venv/bin/python
export REPLAY_OUT=/mnt/fast/scholarsreadinglist/search/fields-r4 REPLAY_RESULTS=replay_r4.json READINGLISTS_RESULTS=readinglists_r4.json REPLAY_ARMS=new2B-only
export SRL_FIELDS_GPU=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=/home/maxime/Projects/scholarsreadinglist/search-dev \
       PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=16 \
       SRL_RELEASE=/mnt/fast/scholarsreadinglist/release/releases/20261016T073008Z \
       SRL_VLLM_PYTHON=/nix/store/skvyvs8jj0k7q4vw2137blrnfp3y5cb2-vllm-runtime-0.29.0/bin/python \
       TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
"$PY" replay.py
"$PY" readinglists.py
echo "$(date -Is) REPLAY AND READING LISTS DONE"
