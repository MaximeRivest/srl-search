#!/usr/bin/env bash
# fieldpick on GPU 0: pauses the public 9B rewriter only while it runs; always restores it on exit.
set -euo pipefail
cd "$(dirname "$0")"
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
until curl -sf -m 5 127.0.0.1:8795/api/status | grep -q '"queued": 0, "running": 0'; do
  echo "$(date -Is) a rewrite is running or queued; waiting"; sleep 30
done
paused=1
systemctl --user stop smr-worker-home
sudo -n systemctl stop model-vllm-smr-9b
sleep 10
echo "$(date -Is) rewriter paused; GPU 0 free"
export SRL_FIELDS_GPU=0 FIELDPICK_NO_GATE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID TRITON_LIBCUDA_PATH=/run/opengl-driver/lib HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
for i in 1 2 3; do ../../../.venv/bin/python fieldpick.py run && exit 0; echo "try $i failed; retrying in 1 min"; sleep 60; done; exit 1
