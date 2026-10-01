#!/bin/bash
# DeepSWE again for one served model, graded on all of the agent's changes (committed or not; see
# deepswe-env/deep-swe-upstream), pi and prime_agent side by side, once run_tb4_after.sh logs "tb4 ... done" or
# "tb4 skipped". Label <label>-wt1. Progress goes to logs/baselines-<label>.log.
#
#   ./run_deepswe_wt_after.sh <label> <local port>
set -uo pipefail
cd "$(dirname "$0")"
LABEL=$1; export PORT=$2
log() { echo "$(date +%T) $*" >> logs/baselines-$LABEL.log; }
until grep -qE "tb4 .* done|tb4 skipped" logs/baselines-$LABEL.log; do sleep 60; done
curl -sf -m 5 "127.0.0.1:$PORT/v1/models" > /dev/null || { log "deepswe-wt skipped: server not up"; exit 1; }
log "deepswe $LABEL-wt1 start"
./run_eval.sh deepswe pi "$LABEL-wt1" 4 > /dev/null & A=$!
./run_eval.sh deepswe prime_agent "$LABEL-wt1" 4 > /dev/null & B=$!
wait $A $B; log "deepswe $LABEL-wt1 done"
