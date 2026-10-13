#!/usr/bin/env bash
# A second, private search API running your changes, beside the public one; then promote them in one step.
#
#   search/dev.sh up        start the dev API (127.0.0.1:8811, tailnet :18811) from the dev worktree
#   search/dev.sh restart   after editing (it loads ~20 GB of indexes: about a minute)
#   search/dev.sh down | status | logs
#   search/dev.sh promote   dev branch -> master, restart the public API, smoke test; back to the old
#                           version by itself if the new one doesn't answer
#   search/dev.sh rollback  the commit before the last promote, restarted
#
# Where to edit: ~/Projects/scholarsreadinglist/search-dev (a git worktree on branch "dev"), never the main
# checkout: the public API (srl-search) runs the main checkout and restarts on its own (nightly data release,
# crashes, reboots), so a half-done edit there would go public. Commit in the worktree, then promote.
# The dev API reads the same data release as the public one (read-only) and uses the same Python environment.
# Point the dev site at it: sciencemadereadable/app/tools/dev.sh up --data dev
# Cost: a second ~20 GB process while it runs, at low CPU priority (the public API keeps first claim).
set -euo pipefail
MAIN="$(cd "$(dirname "$0")" && pwd)"
WT="$(dirname "$MAIN")/search-dev"
PORT=8811
TS_PORT=18811

health() { for _ in $(seq 1 ${2:-90}); do curl -sf -o /dev/null "http://127.0.0.1:$1/health" && return 0; sleep 2; done; return 1; }
smoke() {
  curl -sf "http://127.0.0.1:$1/search?q=coral+reef+bleaching&k=3&map=true&abstracts=false" | grep -q '"results":\[{' &&
  curl -sf "http://127.0.0.1:$1/search?q=%22gut+microbiome%22+autism&k=3&abstracts=false" | grep -q '"mode":"keyword"' &&
  curl -sf "http://127.0.0.1:$1/works/W4410027237" | grep -q '"title"' &&
  curl -sf -X POST "http://127.0.0.1:$1/mcp" -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
    -H 'Mcp-Protocol-Version: 2025-11-25' -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search_papers","arguments":{"query":"coral reef bleaching","limit":1,"abstracts":"none"}}}' \
    | grep -q 'https://doi.org/'
}

case "${1:-status}" in
  up|restart)
    if [ ! -d "$WT" ]; then
      git -C "$MAIN" show-ref -q refs/heads/dev || git -C "$MAIN" branch dev master
      git -C "$MAIN" worktree add -q "$WT" dev
      echo "dev worktree: $WT (branch dev)"
    fi
    systemctl --user stop srl-search-dev 2>/dev/null || true
    systemctl --user reset-failed srl-search-dev 2>/dev/null || true
    # same settings as the public unit (~/Projects/os/machines/hm/scholarsreadinglist-data.nix), lower priority
    systemd-run --user -q --unit=srl-search-dev --description="search API, dev branch (127.0.0.1:$PORT)" \
      -p WorkingDirectory="$WT" -p CPUWeight=100 -p IOWeight=100 -p MemoryMax=40G -p LimitNOFILE=262144 \
      -E TOKENIZERS_PARALLELISM=false -E SRL_TOPICS_CACHE=/mnt/fast/scholarsreadinglist/search/topics-cache-dev \
      -E SRL_SMR=/mnt/fast/scholarsreadinglist/search/smr-2026-10-08 \
      -E SRL_CORPUS=/mnt/fast/scholarsreadinglist/search/corpus-2026-10-05 \
      -E SRL_ATLAS=/mnt/fast/science-atlas/build/20261008T131030-tsne \
      "$MAIN/.venv/bin/python" -m uvicorn srl_search.app:app --host 127.0.0.1 --port "$PORT"
    tailscale serve --bg --https="$TS_PORT" "http://127.0.0.1:$PORT" >/dev/null
    echo "loading indexes…"
    health "$PORT" && echo "dev API: http://127.0.0.1:$PORT  https://lambda.tail69222b.ts.net:$TS_PORT/ (test page)" \
      || { journalctl --user -u srl-search-dev -n 30 -o cat --no-pager; exit 1; }
    ;;
  down)
    systemctl --user stop srl-search-dev 2>/dev/null || true
    tailscale serve --https="$TS_PORT" off >/dev/null 2>&1 || true
    echo "dev API stopped"
    ;;
  status)
    echo "dev API: $(systemctl --user is-active srl-search-dev 2>/dev/null || true)  branch dev at $(git -C "$MAIN" log -1 --format='%h %s' dev 2>/dev/null | cut -c1-70)"
    echo "public:  $(systemctl --user is-active srl-search)  master at $(git -C "$MAIN" log -1 --format='%h %s' master | cut -c1-70)"
    ;;
  logs) journalctl --user -u srl-search-dev -f -o cat ;;
  promote)
    [ -z "$(git -C "$WT" status --porcelain)" ] || { git -C "$WT" status --short; echo "commit in $WT first" >&2; exit 1; }
    [ -z "$(git -C "$MAIN" status --porcelain)" ] || { git -C "$MAIN" status --short; echo "the main checkout has uncommitted changes: someone edited the public copy; sort that out first" >&2; exit 1; }
    (cd "$WT" && "$MAIN/.venv/bin/python" -m pytest -q) || { echo "tests fail on dev: not promoting" >&2; exit 1; }
    OLD=$(git -C "$MAIN" rev-parse HEAD)
    git -C "$MAIN" merge -q --ff-only dev || { echo "master has commits dev doesn't: in $WT run 'git merge master', test, then promote" >&2; exit 1; }
    echo "$OLD" > "$MAIN/.last-promote"
    echo "master $(git -C "$MAIN" log -1 --format='%h %s' | cut -c1-70); restarting the public API (the site falls back to Europe PMC for ~1 min)"
    systemctl --user restart srl-search
    if health 8810 && smoke 8810; then echo "promoted; public API answers"; exit 0; fi
    echo "the new version doesn't answer: going back to ${OLD:0:8}" >&2
    git -C "$MAIN" reset -q --keep "$OLD"
    systemctl --user restart srl-search
    health 8810 && echo "back on ${OLD:0:8}" >&2
    exit 1
    ;;
  rollback)
    OLD=$(cat "$MAIN/.last-promote")
    git -C "$MAIN" reset -q --keep "$OLD"
    systemctl --user restart srl-search
    health 8810 && smoke 8810 && echo "public API back on ${OLD:0:8}"
    ;;
  *) sed -n '2,19p' "$0"; exit 2 ;;
esac
