#!/usr/bin/env bash
# Give GPU 0 back to the rewriter: no new map borrows it (srl_search/judges.py BORROW_STOP), wait up to 45 min for a
# map judging there to finish, then start the 9B (clearing an earlier failed start) and the home worker.
set -uo pipefail
SUDO=/run/wrappers/bin/sudo; [ -x "$SUDO" ] || SUDO=sudo   # NixOS: the setuid one (a service PATH can hold a plain copy)
stop=/run/user/$(id -u)/srl-gpu0-returning
touch "$stop"
trap 'rm -f "$stop"' EXIT
for i in $(seq 1 90); do
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | while IFS=, read g p; do
           [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0)" = "$g" ] || continue
           ps -o args= -p $p | grep -q -E "vllm_decider|srl_search.judges" && echo "$p"; done)
  [ -z "$busy" ] && break
  echo "$(date -Is) a map is judging on GPU 0; waiting before restarting the rewriter"; sleep 30
done
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
if [ "$free" -lt 21000 ]; then      # something else (training, tests) uses GPU 0 now: its owner restarts the rewriter
  echo "$(date -Is) GPU 0 is in use by something else (${free} MB free); not restarting the rewriter"
  exit 0
fi
"$SUDO" -n systemctl reset-failed model-vllm-smr-9b 2>/dev/null || true
"$SUDO" -n systemctl start model-vllm-smr-9b || echo "WARNING: could not restart model-vllm-smr-9b"
systemctl --user start smr-worker-home || echo "WARNING: could not restart smr-worker-home"
echo "$(date -Is) rewriter restored (the 9B takes ~6 minutes to load)"
