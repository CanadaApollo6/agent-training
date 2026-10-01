#!/bin/bash
# TB2 pilot attempts (tasks.txt) for one served build x one harness, as R1s's local series ran them: 4 at a time,
# 60 turns, 1-hour rollouts, one attempt after another (labels <label>-a<N>). Waits up to an hour for the server on
# the local tunnel port, then terminates the pod and logs it. Progress in logs/pod-series.log.
#
#   ATTEMPTS="1 2 3" setsid nohup ./run_pod_series.sh <label, e.g. ornith35b-r1s-sd> <harness> <local port> <pod id> \
#       > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
LABEL=$1; HARNESS=$2; export PORT=$3; POD=$4
log() { echo "$(date +%T) $*" >> logs/pod-series.log; }
up() { curl -sf -m 5 "127.0.0.1:$PORT/v1/models" > /dev/null; }
for i in $(seq 180); do up && break; sleep 20; done
if ! up; then log "$LABEL $HARNESS: server never came up on $PORT; not running"
else
  for a in ${ATTEMPTS:-1 2 3}; do
    log "$LABEL-a$a $HARNESS start (port $PORT, pod ${POD:0:8})"
    TIMEOUT=3600 ./run_pilot.sh "$HARNESS" "$LABEL-a$a" 4 > /dev/null 2>&1
    log "$LABEL-a$a $HARNESS done (exit $?)"
    up || { log "$LABEL $HARNESS: server gone after attempt $a; stopping"; break; }
  done
fi
cd /tmp && timeout 120 uvx prime pods terminate "$POD" --yes > /tmp/terminate-$POD.log 2>&1
echo "$(date -Is) TERMINATE pod ${POD:0:8} (TB2 pilot series $LABEL $HARNESS done) exit=$?" >> /tmp/prime-spend.log
