#!/usr/bin/env bash
# Train and test the round-3 2B on GPU 0: pauses the public 9B rewriter only while it runs; always restores it on exit.
set -uo pipefail
cd "$(dirname "$0")"
M=/mnt/fast/scholarsreadinglist/fieldjudge/v3/models
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
echo "$(date -Is) rewriter paused; GPU 0 free for the 2B"
export FJ3_GPU=0 FJ3_SCALE_MAX=3 FJ3_SIZES=2B
bash run3.sh || { echo "2B training stopped"; exit 1; }
bash evals.sh "v3-2B=$M/2B/merged"
echo "$(date -Is) ROUND 3 2B DONE"
