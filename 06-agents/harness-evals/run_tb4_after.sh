#!/bin/bash
# Terminal-Bench 4 for one served model, pi and prime_agent side by side, once its run_baselines.sh batch says "all done"
# (so the server only ever carries one pair). Progress goes to the same logs/baselines-<label>.log.
#
#   ./run_tb4_after.sh <label> <local port>
set -uo pipefail
cd "$(dirname "$0")"
LABEL=$1; export PORT=$2
log() { echo "$(date +%T) $*" >> logs/baselines-$LABEL.log; }
until grep -qE "all done|stopping|not up" logs/baselines-$LABEL.log; do sleep 60; done
grep -q "all done" logs/baselines-$LABEL.log || { log "tb4 skipped: batch didn't finish"; exit 1; }
log "tb4 $LABEL-a1 start"
./run_eval.sh tb4 pi "$LABEL-a1" 4 > /dev/null & A=$!
./run_eval.sh tb4 prime_agent "$LABEL-a1" 4 > /dev/null & B=$!
wait $A $B; log "tb4 $LABEL-a1 done"
