#!/bin/bash
# R1s head to head: the Python 0.6.0 port (serve_kernels.py new) vs the Zig engine, same prompts, 64K context, one at a time.
# Each: bench_openai.py (2 prompts x T=1/greedy x 3 reps, 256 tokens), longctx.py prefill at ~16K/~40K, long_check.py decode.
set -u
A=~/Projects/Personal/Research/agent-training/01-inference
H=$A/speed-hillclimb/tensorfold; Z=$A/qwen38-27b/zig; OUT=$Z/results/r1s; TF=$A/tools/TensorFold-moe
export TENSORFOLD_NO_UPDATE_CHECK=1 CUDA_HOME=/opt/cuda PATH=/opt/cuda/bin:$PATH
up() { for i in $(seq 1 240); do curl -sf http://127.0.0.1:$1/v1/models > /dev/null && return 0; sleep 2; done; return 1; }
run() {  # tag port model
  echo "== $1"; nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader
  (cd $A/envs/tensorfold060 && uv run python $A/tools/TensorFold-060/tools/bench_openai.py http://127.0.0.1:$2 $3 --tokens 256 --reps 3 --temperatures 1.0,0 --label $1 --output $OUT/h2h-bench-$1.json 2>&1 | tail -5)
  for n in 16000 40000; do MODEL_NAME=$3 python3 $H/longctx.py $2 $n; done
  python3 $Z/long_check.py http://127.0.0.1:$2 40000 $TF/
}
# Python port
cd $A/envs/tensorfold060
TENSORFOLD_CUDA_RESERVE_GIB=1 TF_GPU_ONLY_BUDGET=1 uv run python $H/serve_kernels.py new serve ~/models/ornith/r1s --name ornith --context 65536 --port 8090 --prompt-cache-gib 0 > ~/.cache/r1s-h2h/py.log 2>&1 &
PY=$!
up 8090 || { tail -20 ~/.cache/r1s-h2h/py.log; exit 1; }
run python 8090 ornith
kill $PY; wait $PY 2>/dev/null
for i in $(seq 30); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 1; done
# Zig engine
cd $TF
setsid ./zig-out/native/bin/tensorfold-native serve ~/models/ornith/r1s --context 65536 --parallel 1 --port 8095 --backend cuda --no-update-check > ~/.cache/r1s-h2h/zig.log 2>&1 < /dev/null &
up 8095 || { tail -20 ~/.cache/r1s-h2h/zig.log; exit 1; }
run zig 8095 r1s
for p in $(ps -C tensorfold-nati -o pid=); do kill $p; done
