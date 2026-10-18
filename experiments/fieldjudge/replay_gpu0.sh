#!/usr/bin/env bash
# Free (no reference model, no Jev) replay of the round-2 judges on GPU 0, then the reading-list check.
# Pauses the public 9B rewriter only while it runs; always restores it on exit (success, failure or stop).
# Does not start InkType. The reference model is used only afterwards, to SCORE reading lists (~$1), never inside the maps.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/maxime/Projects/scholarsreadinglist/search-dev/.venv/bin/python
paused=0
restore() {
  if [[ $paused == 1 ]]; then
    sudo -n systemctl start model-vllm-smr-9b || echo "WARNING: could not restart model-vllm-smr-9b"
    systemctl --user start smr-worker-home || echo "WARNING: could not restart smr-worker-home"
    echo "$(date -Is) rewriter restored (the 9B takes ~6 minutes to load)"
  fi
}
trap restore EXIT
trap 'exit 143' TERM INT
# never cut a rewrite in progress
until curl -sf -m 5 127.0.0.1:8795/api/status | grep -q '"queued": 0, "running": 0'; do
  echo "$(date -Is) a rewrite is running or queued; waiting"; sleep 30
done
paused=1
systemctl --user stop smr-worker-home
sudo -n systemctl stop model-vllm-smr-9b
echo "$(date -Is) rewriter paused; GPU 0 free for the replay"
export REPLAY_OUT=/mnt/fast/scholarsreadinglist/search/fields-r4 REPLAY_RESULTS=replay_r4.json READINGLISTS_RESULTS=readinglists_r4.json REPLAY_ARMS=new08-only
export SRL_FIELDS_GPU=0 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=/home/maxime/Projects/scholarsreadinglist/search-dev \
       PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=16 \
       SRL_RELEASE=/mnt/fast/scholarsreadinglist/release/releases/20261016T073008Z \
       SRL_VLLM_PYTHON=/nix/store/skvyvs8jj0k7q4vw2137blrnfp3y5cb2-vllm-runtime-0.29.0/bin/python \
       TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
"$PY" replay.py
"$PY" readinglists.py
echo "$(date -Is) REPLAY AND READING LISTS DONE"
