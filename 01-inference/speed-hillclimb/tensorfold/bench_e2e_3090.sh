#!/bin/bash
# usage: bench_e2e_3090.sh STEP TAG [extra serve args]
# Serves the MLX 4-bit build on the local 3090 with serve_kernels.py STEP (old/swap/select/new), runs bench_openai.py
# and greedy.py against it, then stops it. The server runs in a 14 GB memory cgroup. Weights dir: MODEL (default mlx4).
set -u
STEP=$1; TAG=$2; shift 2
cd ~/Projects/Personal/Research/agent-training/01-inference/envs/tensorfold
export TENSORFOLD_CUDA_RESERVE_GIB=${TENSORFOLD_CUDA_RESERVE_GIB:-1} CUDA_HOME=/opt/cuda PATH=/opt/cuda/bin:$PATH TENSORFOLD_NO_UPDATE_CHECK=1
OUT=~/Projects/Personal/Research/agent-training/01-inference/speed-hillclimb/tensorfold/results
H=~/Projects/Personal/Research/agent-training/01-inference/speed-hillclimb/tensorfold
TF_GPU_ONLY_BUDGET=1 systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=0 uv run python $H/serve_kernels.py $STEP serve ${MODEL:-$HOME/models/ornith/mlx4} --name ornith --context 16384 --port 8090 --prompt-cache-gib 0 "$@" > /tmp/serve-$TAG.log 2>&1 &
PID=$!
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8090/v1/models > /dev/null && break
  kill -0 $PID 2>/dev/null || { echo "server died"; tail -20 /tmp/serve-$TAG.log; exit 1; }
  sleep 2
done
echo "up after $((i*2)) s"; nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader
uv run python ../../tools/TensorFold/tools/bench_openai.py http://127.0.0.1:8090 ornith --tokens 256 --reps 3 --temperatures 1.0,0 --label $TAG --output $OUT/bench-3090-$TAG.json 2>&1 | tail -6
uv run python $H/greedy.py 8090 $OUT/greedy-3090-$TAG.json 2>&1 | tail -4
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader
kill $PID; wait $PID 2>/dev/null
# uv's child python: make sure the port's server is gone
for p in $(pgrep -f "python .*serve_kernels.py $STEP serve"); do [ "$p" != "$$" ] && kill $p 2>/dev/null; done
sleep 2; nvidia-smi --query-gpu=memory.used --format=csv,noheader
