#!/bin/bash
# Self-distillation round-2 data: prime_agent rollouts from one pod's servers on TB2 training tasks (never the pilot).
# Each <shard>:<local port> pair runs datagen/<shard>.txt (tasks solved before) 8 times each, then datagen/<shard>x.txt
# (never solved) twice each, 4 at a time, at the eval caps (60 turns, 1 h). Shards on one pod run in parallel; when
# all are done the pod is terminated and logged. Labels ornith35b-r1s-sd-dg-<shard>[x]; progress in logs/datagen.log.
#
#   setsid nohup ./run_datagen.sh <pod id> s1:8131 [s2:8132 ...] > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
POD=$1; shift
log() { echo "$(date +%T) $*" >> logs/datagen.log; }
up() { curl -sf -m 5 "127.0.0.1:$1/v1/models" > /dev/null; }
shard() {
  local S=$1 PORT=$2
  for i in $(seq 180); do up "$PORT" && break; sleep 20; done
  up "$PORT" || { log "$S: server never came up on $PORT"; return; }
  log "$S start (port $PORT, pod ${POD:0:8})"
  PORT=$PORT ROLLOUTS=8 TASKS_FILE=datagen/$S.txt ./run_eval.sh tb2 prime_agent "ornith35b-r1s-sd-dg-$S" 4 > /dev/null 2>&1
  log "$S solved-before tasks done (exit $?)"
  up "$PORT" || { log "$S: server gone; skipping never-solved"; return; }
  PORT=$PORT ROLLOUTS=2 TASKS_FILE=datagen/${S}x.txt ./run_eval.sh tb2 prime_agent "ornith35b-r1s-sd-dg-${S}x" 4 > /dev/null 2>&1
  log "$S never-solved tasks done (exit $?)"
}
for spec in "$@"; do shard "${spec%%:*}" "${spec##*:}" & done
wait
cd /tmp && timeout 120 uvx prime pods terminate "$POD" --yes > /tmp/terminate-$POD.log 2>&1
echo "$(date -Is) TERMINATE pod ${POD:0:8} (round-2 datagen $* done) exit=$?" >> /tmp/prime-spend.log
