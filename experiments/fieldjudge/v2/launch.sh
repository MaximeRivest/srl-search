#!/usr/bin/env bash
# Independent user-systemd processes, not children of the conversation.
# Run again to resume from checkpoints. Use --restart to reload changed code.
# Temporary failures (75) retry after five minutes; safety/data pauses (78) do not.
# These are transient jobs: survive closing Chattering/logout (with user linger),
# not a reboot. After reboot, run this script again.
set -euo pipefail
cd "$(dirname "$0")"
units=(fj-v2-prepare fj-v2-first-sol fj-v2-first-opus)
if [[ ${1:-} == --restart ]]; then
  systemctl --user stop "${units[@]}" 2>/dev/null || true
fi
start_job() {
  local unit=$1 script=$2
  shift 2
  local state
  state=$(systemctl --user show "$unit.service" --property=ActiveState --value 2>/dev/null || true)
  if [[ "$state" == active || "$state" == activating ]]; then
    echo "$unit already active; leaving it alone"
    return
  fi
  systemctl --user reset-failed "$unit" 2>/dev/null || true
  systemd-run --user --unit="$unit" --description="Field judge v2: $script $*" \
    -p WorkingDirectory="$PWD" -p Nice=10 -p CPUWeight=30 -p MemoryMax=60G -p LimitNOFILE=262144 \
    -p Restart=on-failure -p RestartSec=300 -p RestartPreventExitStatus=78 \
    -p RemainAfterExit=yes -p StartLimitIntervalSec=3600 -p StartLimitBurst=8 \
    -p TimeoutStopSec=30 -p KillMode=control-group \
    -E PYTHONUNBUFFERED=1 /bin/bash "$script" "$@"
}
start_job fj-v2-prepare prepare.sh
start_job fj-v2-first-sol first_cohort.sh sol
start_job fj-v2-first-opus first_cohort.sh opus
printf '\nProgress: journalctl --user -u fj-v2-prepare -f\n'
