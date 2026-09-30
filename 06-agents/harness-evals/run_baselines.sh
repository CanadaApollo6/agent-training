#!/bin/bash
# Untrained harness baselines for one served model: pi and prime_agent side by side on the same server (4 tasks each),
# TB2 attempt 1, then DeepSWE (25 tasks, 1 attempt), then TB2 attempts 2 and 3. Progress in logs/baselines-<label>.log.
#
#   ./run_baselines.sh <label, e.g. qwen38-27b-q8h> <local port of the tunnel to its llama-server>
set -uo pipefail
cd "$(dirname "$0")"
LABEL=$1; export PORT=$2
log() { echo "$(date +%T) $*" >> logs/baselines-$LABEL.log; }
up() { curl -sf -m 10 127.0.0.1:$PORT/v1/models >/dev/null; }
pair() {                       # suite, label: both harnesses at once
  log "$1 $2 start"
  ./run_eval.sh "$1" pi "$2" 4 > /dev/null & A=$!
  ./run_eval.sh "$1" prime_agent "$2" 4 > /dev/null & B=$!
  wait $A $B; log "$1 $2 done"
  up || { log "server gone after $1 $2; stopping"; exit 1; }
}
up || { log "server not up"; exit 1; }
pair tb2 $LABEL-a1
pair deepswe $LABEL-a1
pair tb2 $LABEL-a2
pair tb2 $LABEL-a3
log "all done"
