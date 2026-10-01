#!/bin/bash
# Did Ornith 1.5's Claude Code training survive compression? One model under the claude_code harness on DeepSWE (the
# 10 tasks, all-changes grading; 100 turns, 90 minutes, 4 at a time, as the pi and prime_agent DeepSWE runs), served
# with a /v1/messages endpoint: R1s through TensorFold's own (claude-code.patch), bf16 through vLLM and
# messages_proxy.py (same translation). Waits up to an hour for the server, then terminates the pod and logs it.
#   setsid nohup ./run_claude_code.sh <label> <local port> <pod id> > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
LABEL=$1; export PORT=$2; POD=$3
log() { echo "$(date +%T) $*" >> logs/claude-code.log; }
for i in $(seq 180); do curl -sf -m 5 "127.0.0.1:$PORT/v1/models" > /dev/null && break; sleep 20; done
if ! curl -sf -m 5 "127.0.0.1:$PORT/v1/models" > /dev/null; then log "$LABEL: server never came up on $PORT; not running"
else
log "deepswe $LABEL claude_code start (port $PORT, pod ${POD:0:8})"
./run_eval.sh deepswe claude_code "$LABEL" 4 > /dev/null
log "deepswe $LABEL claude_code done (exit $?)"
fi
cd /tmp && timeout 120 uvx prime pods terminate "$POD" --yes > /tmp/terminate-$POD.log 2>&1
echo "$(date -Is) TERMINATE pod ${POD:0:8} (Claude Code DeepSWE $LABEL done) exit=$?" >> /tmp/prime-spend.log
