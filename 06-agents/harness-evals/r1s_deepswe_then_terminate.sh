#!/bin/bash
# After an R1s full-TB2 half exits, run R1s on DeepSWE (all-changes grading) with the same harness on the same pod,
# then terminate the pod and log it in /tmp/prime-spend.log.
#   setsid nohup ./r1s_deepswe_then_terminate.sh <pid> <pi|prime_agent> <local port> <pod id> &
set -uo pipefail
cd "$(dirname "$0")"
PID=$1; H=$2; export PORT=$3; POD=$4
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "$(date +%T) deepswe ornith35b-r1s-wt1 $H start" >> logs/r1s-full.log
curl -sf -m 5 "127.0.0.1:$PORT/v1/models" > /dev/null && ./run_eval.sh deepswe "$H" ornith35b-r1s-wt1 4 > /dev/null
echo "$(date +%T) deepswe ornith35b-r1s-wt1 $H done" >> logs/r1s-full.log
cd /tmp && timeout 120 uvx prime pods terminate "$POD" --yes > /tmp/terminate-$POD.log 2>&1
echo "$(date -Is) TERMINATE pod ${POD:0:8} (R1s TB2 full + DeepSWE $H done) exit=$?" >> /tmp/prime-spend.log
