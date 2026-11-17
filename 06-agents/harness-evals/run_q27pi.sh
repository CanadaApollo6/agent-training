#!/bin/bash
# harness-hub pi adapter test, local side: tunnels to the three servers of serve_qwen27_pod.sh, then 3 attempts per arm,
# all 9 side by side (run_eval.sh tb2 under pi: tasks.txt, 60 turns, 60 min, -c 7, 32K reply cap for every arm).
# Terminates the pod when the last attempt ends. Progress in logs/q27pi-series.log, pod events in
# results/q27pi-pods.log.
#
#   setsid nohup ./run_q27pi.sh <label prefix> <pod ip> <pod id> > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
P=$1; IP=$2; POD=$3
log() { echo "$(date +%T) $*" >> logs/q27pi-series.log; }
POD_USER=root setsid nohup ./tunnels.sh 8301:$IP:8000 8302:$IP:8001 8303:$IP:8002 > /dev/null 2>&1 < /dev/null &
TUN=$!
up() { curl -sf -m 5 "127.0.0.1:$1/v1/models" > /dev/null; }
for i in $(seq 180); do up 8301 && up 8302 && up 8303 && break; sleep 20; done
if ! { up 8301 && up 8302 && up 8303; }; then log "$P: servers never all came up; not running"
else
  PIDS=()
  for arm in med:8301 off:8302 ad:8303; do
    for a in 1 2 3; do
      L=$P-${arm%%:*}-a$a
      ( log "$L start"
        PORT=${arm##*:} EXTRA="--sampling.max-tokens 32768" ./run_eval.sh tb2 pi "$L" 7 > /dev/null 2>&1
        log "$L done (exit $?): $(grep -c 'reward=1' logs/tb2-$L-pi.log) solved" ) &
      PIDS+=($!)
      sleep 20
    done
  done
  wait "${PIDS[@]}"
fi
kill $TUN 2> /dev/null
cd /tmp && timeout 120 uvx prime pods terminate "$POD" --yes > /tmp/terminate-$POD.log 2>&1
echo "$(date '+%F %T') terminate q27pi-adapter (series done) exit=$?" >> "$OLDPWD/results/q27pi-pods.log"
echo "$(date -Is) TERMINATE pod ${POD:0:8} (q27pi series done)" >> /tmp/prime-spend.log
