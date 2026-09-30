#!/bin/bash
# R1s (R1 with the always-on path re-rounded by the imatrix search, see 05-compression/ornith/kl) on the local 3090,
# same settings as R1's series: swap 8 GiB, 3 attempts 4 at a time, 10-minute cool-downs. Progress in
# logs/r1s-series.log (ATTEMPTS="2 3" to run only some), GPU temperature and power each minute in logs/r1s-gpu.log.
set -uo pipefail
cd "$(dirname "$0")"
log() { echo "$(date +%T) $*" >> logs/r1s-series.log; }
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader -l 60 >> logs/r1s-gpu.log &
GPU=$!
MODEL=~/models/ornith/r1s NAME=ornith35b-r1s SWAP_GIB=8 bash serve_r2_local.sh >> logs/r1s-series.log 2>&1 || { log "server failed"; kill $GPU; exit 1; }
up() { curl -sf -m 5 127.0.0.1:8000/v1/models >/dev/null; }
for a in ${ATTEMPTS:-1 2 3}; do
  log "attempt $a start"; PORT=8000 TIMEOUT=3600 ./run_pilot.sh pi ornith35b-r1s-c4s-a$a 4; log "attempt $a done"
  up || { log "server gone after attempt $a: its results need checking; stopping"; kill $GPU; exit 1; }
  [ $a -lt 3 ] && sleep 600
done
P=$(ss -ltnp | grep ':8000 ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2); [ -n "$P" ] && kill "$P"
kill $GPU
log "series done; server stopped (pid $P)"
