#!/bin/bash
# Serve the R2 build (gate/up 2-bit, down 3-bit experts) locally with the patched TensorFold on 127.0.0.1:8000, a fresh
# sampling seed per request, 131K context; log in /tmp/serve-r2-evals.log. Stop it by the pid listening on :8000.
set -euo pipefail
cd "$(dirname "$0")/../../01-inference/envs/tensorfold"
H=../../speed-hillclimb/tensorfold
TMPDIR=$HOME/.cache/tf-tmp TENSORFOLD_NO_UPDATE_CHECK=1 CUDA_HOME=/opt/cuda PATH=/opt/cuda/bin:$PATH \
  TENSORFOLD_CUDA_RESERVE_GIB=0.3 TF_GPU_ONLY_BUDGET=1 TF_RANDOM_SEED=1 \
  nohup systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=0 uv run python $H/serve_kernels.py new serve \
  "${MODEL:-$HOME/models/ornith/r2}" --name ornith35b-r2 --context 131072 --port 8000 --prompt-cache-gib 0 \
  > /tmp/serve-r2-evals.log 2>&1 &
for i in $(seq 100); do curl -sf -m 2 127.0.0.1:8000/v1/models >/dev/null && { echo up; exit 0; }; sleep 3; done
echo "server didn't come up"; tail -20 /tmp/serve-r2-evals.log; exit 1
