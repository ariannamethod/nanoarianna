#!/bin/bash
# budget_watchdog.sh — runs on phone-2 via cron, kills the RunPod pod
# if cumulative cost exceeds the cap. Per Phase 4 brief §"Pre-flight"
# item 6 + §"Open decisions" §5.
#
# Cron entry (added at pod-launch step):
#   */5 * * * * /data/data/com.termux/files/home/nanoarianna/runpod/budget_watchdog.sh >> ~/.nanoarianna/watchdog.log 2>&1
#
# Configuration: edit the three variables below or override via env at
# install time (`POD_ID=xyz HARD_KILL_USD=18 RATE_USD_HR=2.79 ./budget_watchdog.sh`).

set -u

POD_ID="${POD_ID:-}"
HARD_KILL_USD="${HARD_KILL_USD:-18}"
SOFT_WARN_USD="${SOFT_WARN_USD:-14}"
RATE_USD_HR="${RATE_USD_HR:-2.79}"   # A100 80GB SXM ~ $2.79/hr per RunPod template

LOGDIR="${HOME}/.nanoarianna"
mkdir -p "$LOGDIR"

if [ -z "$POD_ID" ]; then
    # No pod active — read from state file (written at pod-launch step)
    if [ -f "$LOGDIR/active_pod.id" ]; then
        POD_ID=$(cat "$LOGDIR/active_pod.id")
    else
        # No active pod — exit silently, watchdog has nothing to do
        exit 0
    fi
fi

TOKEN="$(cat ~/.config/runpod/token 2>/dev/null)"
if [ -z "$TOKEN" ]; then
    echo "[watchdog $(date +%FT%T)] ERROR: ~/.config/runpod/token missing" >&2
    exit 2
fi

# Query pod runtime via RunPod GraphQL.
QUERY='{"query":"query Pod($input: PodFilter){ pod(input: $input) { id runtimeMinutes desiredStatus } }","variables":{"input":{"podId":"'"$POD_ID"'"}}}'

RESP=$(curl -sS -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
    -d "$QUERY" https://api.runpod.io/graphql)

MINUTES=$(echo "$RESP" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    rt = d.get('data', {}).get('pod', {}).get('runtimeMinutes', 0)
    print(rt if rt is not None else 0)
except Exception as e:
    print('0', file=sys.stderr)
    print(0)
")

# Awk float math (no bc dependency on Termux)
COST=$(awk -v m="$MINUTES" -v r="$RATE_USD_HR" 'BEGIN{printf "%.2f", m/60.0*r}')

NOW=$(date +%FT%T)
echo "[watchdog $NOW] pod=$POD_ID minutes=$MINUTES cost=\$$COST cap=\$$HARD_KILL_USD"

# Warn at soft, kill at hard
if awk -v c="$COST" -v cap="$HARD_KILL_USD" 'BEGIN{exit !(c >= cap)}'; then
    echo "[watchdog $NOW] HARD KILL — cost \$$COST >= cap \$$HARD_KILL_USD" >&2
    KILL='{"query":"mutation StopPod($input: PodStopInput!){ podStop(input: $input) { id desiredStatus } }","variables":{"input":{"podId":"'"$POD_ID"'"}}}'
    curl -sS -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
        -d "$KILL" https://api.runpod.io/graphql
    echo
    rm -f "$LOGDIR/active_pod.id"
    exit 0
elif awk -v c="$COST" -v warn="$SOFT_WARN_USD" 'BEGIN{exit !(c >= warn)}'; then
    echo "[watchdog $NOW] SOFT WARN — cost \$$COST >= warn \$$SOFT_WARN_USD (cap \$$HARD_KILL_USD)" >&2
fi
