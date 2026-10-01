#!/bin/bash
# Does a bigger budget help prime_agent more than pi? Rerun one harness's budget-limited TB2 failures (max_turns or
# timeout in ornith35b-r1s-full-a1, listed in tb2_budget_<harness>.txt by sort_failures.py) with 200 turns and a 4-hour
# rollout timeout (Ornith's TB2.1 setting), 4 at a time like the original, then terminate the pod and log it.
#   setsid nohup ./run_bigbudget.sh <pi|prime_agent> <local port> <pod id> > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
H=$1; export PORT=$2; POD=$3
log() { echo "$(date +%T) $*" >> logs/r1s-bigbudget.log; }
for i in $(seq 180); do curl -sf -m 5 "127.0.0.1:$PORT/v1/models" > /dev/null && break; sleep 20; done
if ! curl -sf -m 5 "127.0.0.1:$PORT/v1/models" > /dev/null; then log "$H: server never came up on $PORT; not running"
else
log "tb2full ornith35b-r1s-big-a1 $H start (port $PORT, pod ${POD:0:8})"
TASKS_FILE=tb2_budget_$H.txt TURNS=200 TIMEOUT=14400 ./run_eval.sh tb2full "$H" ornith35b-r1s-big-a1 4 > /dev/null
log "tb2full ornith35b-r1s-big-a1 $H done (exit $?)"
fi
cd /tmp && timeout 120 uvx prime pods terminate "$POD" --yes > /tmp/terminate-$POD.log 2>&1
echo "$(date -Is) TERMINATE pod ${POD:0:8} (R1s budget-failure rerun $H done) exit=$?" >> /tmp/prime-spend.log
