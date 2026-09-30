#!/bin/bash
# cache_check.py end to end: serve R1, the live pass, stop the server, the cold pass in-process. GPU temperature is
# logged every 30 s to /tmp/cache-temp.log; logs in /tmp/cache-{live,cold}.log.
cd "$(dirname "$0")"
while sleep 30; do nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader; done > /tmp/cache-temp.log 2>&1 &
TEMP=$!
MODEL=$HOME/models/ornith/r1 NAME=ornith35b-r1 SWAP_GIB=8 ./serve_r2_local.sh || { kill $TEMP; exit 1; }
timeout 900 uv run python cache_check.py live > /tmp/cache-live.log 2>&1
echo "live exit $?"; tail -3 /tmp/cache-live.log
kill "$(ss -ltnp | grep ':8000' | grep -o 'pid=[0-9]*' | cut -d= -f2)"; sleep 8
cd ../../01-inference/envs/tensorfold
TMPDIR=$HOME/.cache/tf-tmp TENSORFOLD_NO_UPDATE_CHECK=1 CUDA_HOME=/opt/cuda PATH=/opt/cuda/bin:$PATH \
  TENSORFOLD_CUDA_RESERVE_GIB=0.3 timeout 900 systemd-run --user --scope -q -p MemoryMax=22G -p MemorySwapMax=0 \
  uv run python ../../../06-agents/harness-evals/cache_check.py cold > /tmp/cache-cold.log 2>&1
echo "cold exit $?"; tail -45 /tmp/cache-cold.log
kill $TEMP; echo "peak temp $(sort -n /tmp/cache-temp.log | tail -1)"; date +%H:%M
