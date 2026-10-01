#!/bin/bash
# One served model under the claude_code harness, served with a /v1/messages endpoint: R1s through TensorFold's own
# (claude-code.patch), bf16 through vLLM and messages_proxy.py (same translation). SUITE picks the run_eval.sh suite
# (default deepswe: the 10 tasks, all-changes grading, 100 turns, 90 minutes, as the pi and prime_agent DeepSWE runs;
# tb2full: Terminal-Bench 2 at 60 turns, 1 hour, as the pi and prime_agent full runs; TASKS_FILE narrows it). 4 at a
# time. Waits up to an hour for the server, then terminates the pod and logs it.
#   SUITE=tb2full TASKS_FILE=tb2_full_half1.txt setsid nohup ./run_claude_code.sh <label> <local port> <pod id> > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
LABEL=$1; export PORT=$2; POD=$3; SUITE=${SUITE:-deepswe}
log() { echo "$(date +%T) $*" >> logs/claude-code.log; }
for i in $(seq 180); do curl -sf -m 5 "127.0.0.1:$PORT/v1/models" > /dev/null && break; sleep 20; done
if ! curl -sf -m 5 "127.0.0.1:$PORT/v1/models" > /dev/null; then log "$LABEL: server never came up on $PORT; not running"
else
log "$SUITE $LABEL claude_code start (port $PORT, pod ${POD:0:8}${TASKS_FILE:+, $TASKS_FILE})"
./run_eval.sh "$SUITE" claude_code "$LABEL" 4 > /dev/null
log "$SUITE $LABEL claude_code done (exit $?)"
fi
cd /tmp && timeout 120 uvx prime pods terminate "$POD" --yes > /tmp/terminate-$POD.log 2>&1
echo "$(date -Is) TERMINATE pod ${POD:0:8} (Claude Code $SUITE $LABEL done) exit=$?" >> /tmp/prime-spend.log
