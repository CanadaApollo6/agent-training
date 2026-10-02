#!/bin/bash
# Serve R1s on a Prime pod (A6000, CUDA 13.0 toolkit from NVIDIA's apt repo, uv) the way serve_r2_local.sh does on the
# 3090: patched TensorFold, a fresh sampling seed per request, 131K context, 8 GiB swap, on 127.0.0.1:8080 for an SSH
# tunnel (CONTEXT overrides the 131K window). Expects ~/at/01-inference/{tools/TensorFold,envs/tensorfold,speed-hillclimb/tensorfold/serve_kernels.py}
# and the model in ~/models/ornith/r1s (setup_r1s_pod.sh installs the rest). Log in /tmp/serve-$PORT.log.
# MODEL_DIR and NAME serve another build (e.g. ~/models/ornith/r1s-sd as ornith35b-r1s-sd).
# PORT (default 8080) and GPU (a CUDA device index) run several servers on a multi-GPU pod; logs in /tmp/serve-$PORT.log.
# TF_PASS_UNKNOWN_TOOLS=1 (unknown-tools.patch): calls to tools the request did not declare come back as calls.
set -euo pipefail
export PATH=/usr/local/cuda-13.0/bin:$HOME/.local/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0
PORT=${PORT:-8080}
[ -d /usr/local/cuda-13.0/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES=$GPU
cd ~/at/01-inference/envs/tensorfold
uv sync -q
TENSORFOLD_NO_UPDATE_CHECK=1 TENSORFOLD_CUDA_RESERVE_GIB=0.3 TF_GPU_ONLY_BUDGET=1 TF_RANDOM_SEED=1 TF_PASS_UNKNOWN_TOOLS=1 \
  setsid nohup uv run python ../../speed-hillclimb/tensorfold/serve_kernels.py new serve ${MODEL_DIR:-$HOME/models/ornith/r1s} \
  --name ${NAME:-ornith35b-r1s} --context ${CONTEXT:-131072} --port $PORT --prompt-cache-gib 0 --swap-gib 8 > /tmp/serve-$PORT.log 2>&1 < /dev/null &
for i in $(seq 300); do curl -sf -m 2 127.0.0.1:$PORT/v1/models >/dev/null && { echo up; exit 0; }; sleep 3; done
echo "server didn't come up"; tail -20 /tmp/serve-$PORT.log; exit 1
