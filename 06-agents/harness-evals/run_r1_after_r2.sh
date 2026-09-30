#!/bin/bash
# After R2's third attempt: stop R2, cool down, build R1 (all routed experts 3-bit) on the 3090, cool down, serve it
# with the swap and run 3 attempts 4 at a time with 10-minute cool-downs. Progress in logs/r1-series.log.
set -uo pipefail
cd "$(dirname "$0")"
log() { echo "$(date +%T) $*" >> logs/r1-series.log; }
until grep -q "attempt 3 done" logs/r2c4s-series.log; do sleep 60; done
P=$(ss -ltnp | grep ':8000 ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2); [ -n "$P" ] && kill "$P"
log "R2 done, server stopped; cool-down"; sleep 600
if [ ! -f ~/models/ornith/r1/quantize-report.json ]; then
  log "building R1"
  S=$(ls -d ~/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B/snapshots/*)
  (cd ../../01-inference/envs/tensorfold && CUDA_HOME=/opt/cuda PATH=/opt/cuda/bin:$PATH uv run python \
    ../../speed-hillclimb/tensorfold/quantize_experts.py --bf16 "$S" --base ~/models/ornith/mlx4 \
    --imatrix ~/models/ornith/Ornith-1.5-35B-A3B-imatrix.gguf --out ~/models/ornith/r1 --gate-up 3 --down 3) \
    > /tmp/q-r1.log 2>&1 || { log "R1 build failed (see /tmp/q-r1.log)"; exit 1; }
  log "R1 built ($(du -sh ~/models/ornith/r1 | cut -f1)); cool-down"; sleep 600
fi
MODEL=~/models/ornith/r1 NAME=ornith35b-r1 SWAP_GIB=8 bash serve_r2_local.sh >> logs/r1-series.log 2>&1 || { log "R1 server failed"; exit 1; }
for a in 1 2 3; do
  log "attempt $a start"; PORT=8000 TIMEOUT=3600 ./run_pilot.sh pi ornith35b-r1-c4s-a$a 4; log "attempt $a done"
  [ $a -lt 3 ] && sleep 600
done
log "series done"
