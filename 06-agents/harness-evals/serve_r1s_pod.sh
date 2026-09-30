#!/bin/bash
# Serve R1s on a Prime pod (A6000, CUDA 13.0 toolkit from NVIDIA's apt repo, uv) the way serve_r2_local.sh does on the
# 3090: patched TensorFold, a fresh sampling seed per request, 131K context, 8 GiB swap, on 127.0.0.1:8080 for an SSH
# tunnel. Expects ~/at/01-inference/{tools/TensorFold,envs/tensorfold,speed-hillclimb/tensorfold/serve_kernels.py}
# and the model in ~/models/ornith/r1s (setup_r1s_pod.sh installs the rest). Log in /tmp/serve.log.
# TF_PASS_UNKNOWN_TOOLS=1 (unknown-tools.patch): calls to tools the request did not declare come back as calls.
set -euo pipefail
export PATH=/usr/local/cuda-13.0/bin:$HOME/.local/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0
cd ~/at/01-inference/envs/tensorfold
uv sync -q
TENSORFOLD_NO_UPDATE_CHECK=1 TENSORFOLD_CUDA_RESERVE_GIB=0.3 TF_GPU_ONLY_BUDGET=1 TF_RANDOM_SEED=1 TF_PASS_UNKNOWN_TOOLS=1 \
  setsid nohup uv run python ../../speed-hillclimb/tensorfold/serve_kernels.py new serve ~/models/ornith/r1s \
  --name ornith35b-r1s --context 131072 --port 8080 --prompt-cache-gib 0 --swap-gib 8 > /tmp/serve.log 2>&1 < /dev/null &
for i in $(seq 300); do curl -sf -m 2 127.0.0.1:8080/v1/models >/dev/null && { echo up; exit 0; }; sleep 3; done
echo "server didn't come up"; tail -20 /tmp/serve.log; exit 1
