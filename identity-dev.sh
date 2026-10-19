#!/usr/bin/env bash
# A private search API running the identity branch (paper versions, author identity, rescued abstracts) beside the
# public one, for checking before anything is promoted. 127.0.0.1:8812, tailnet :18812.
#   ./identity-dev.sh up [RELEASE_DIR] | down | logs
# RELEASE_DIR: a release directory (default: the public current one; a test release from release-identity/).
set -euo pipefail
WT="$(cd "$(dirname "$0")" && pwd)"
PORT=8812; TS_PORT=18812
ID=/mnt/fast/scholarsreadinglist/identity/openalex-2026-09-23
case "${1:-up}" in
  up)
    REL="${2:-/mnt/fast/scholarsreadinglist/release/current}"
    systemctl --user stop srl-search-identity 2>/dev/null || true
    systemctl --user reset-failed srl-search-identity 2>/dev/null || true
    systemd-run --user -q --unit=srl-search-identity --description="search API, identity branch (127.0.0.1:$PORT)" \
      -p WorkingDirectory="$WT" -p CPUWeight=100 -p IOWeight=100 -p MemoryMax=40G -p LimitNOFILE=262144 \
      -E TOKENIZERS_PARALLELISM=false -E SRL_RELEASE="$REL" \
      -E SRL_TOPICS_CACHE=/mnt/fast/scholarsreadinglist/search/topics-cache-identity \
      -E SRL_FIELDS=/mnt/fast/scholarsreadinglist/search/fields-identity \
      -E SRL_PEOPLE_CACHE=/mnt/fast/scholarsreadinglist/search/people-cache-identity \
      -E SRL_VERSIONS=$ID/versions -E SRL_IDENTITY=$ID/authors \
      "$WT/.venv/bin/python" -m uvicorn srl_search.app:app --host 127.0.0.1 --port "$PORT"
    tailscale serve --bg --https="$TS_PORT" "http://127.0.0.1:$PORT" >/dev/null 2>&1 || true
    for _ in $(seq 1 120); do curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { echo "up: http://127.0.0.1:$PORT"; exit 0; }; sleep 3; done
    journalctl --user -u srl-search-identity -n 40 -o cat --no-pager; exit 1 ;;
  down) systemctl --user stop srl-search-identity || true; tailscale serve --https="$TS_PORT" off >/dev/null 2>&1 || true ;;
  logs) journalctl --user -u srl-search-identity -n 80 -o cat --no-pager ;;
esac
