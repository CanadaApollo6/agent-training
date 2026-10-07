#!/bin/bash
# usage: bench_tf.sh TAG [extra tensorfold serve args]
# Serves Qwen3.8-27B (MODEL, default the MLX 4-bit build) with TensorFold on the local 3090 and measures decode speed
# with TensorFold's public bench (bench_openai.py: code + chat prompt, greedy and sampled), then stops it.
# LONG adds long-prompt checks. Same guards as speed-hillclimb/tensorfold/bench_e2e_3090.sh: GPU-only memory budget, 14 GB host-RAM cgroup.
set -u
TAG=$1; shift
cd ~/Projects/Personal/Research/agent-training/01-inference/envs/tensorfold
export TENSORFOLD_CUDA_RESERVE_GIB=${TENSORFOLD_CUDA_RESERVE_GIB:-1} CUDA_HOME=/opt/cuda PATH=/opt/cuda/bin:$PATH TENSORFOLD_NO_UPDATE_CHECK=1
H=~/Projects/Personal/Research/agent-training/01-inference
OUT=$H/qwen38-27b/results; LOG=$OUT/serve-$TAG.log
TF_GPU_ONLY_BUDGET=1 systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=0 uv run python $H/speed-hillclimb/tensorfold/serve_kernels.py new serve ${MODEL:-Vontra/Qwen3.8-27B-MLX-4bit} --name qwen27 --port 8091 --prompt-cache-gib 0 "$@" > $LOG 2>&1 &
PID=$!
for i in $(seq 1 300); do
  curl -sf http://127.0.0.1:8091/v1/models > /dev/null && break
  kill -0 $PID 2>/dev/null || { echo "server died"; tail -30 $LOG; exit 1; }
  sleep 2
done
echo "up after $((i*2)) s"; nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader
grep -i -E 'context|kv|budget|GiB|draft' $LOG | head -20
uv run python $H/tools/TensorFold/tools/bench_openai.py http://127.0.0.1:8091 qwen27 --tokens ${TOKENS:-256} --reps 3 --temperatures 1.0,0 --label $TAG --output $OUT/bench-$TAG.json 2>&1 | tail -8
nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader
# LONG="16000 48000": long prompts (recall check), peak VRAM sampled each second
for n in ${LONG:-}; do
  ( while sleep 1; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done ) > /tmp/vram-$TAG-$n.txt &
  VP=$!
  MODEL_NAME=qwen27 uv run python $H/speed-hillclimb/tensorfold/longctx.py 8091 $n 2>&1 | tail -3
  kill $VP; echo "  peak VRAM at ~$n tokens: $(sort -n /tmp/vram-$TAG-$n.txt | tail -1) MiB"
  kill -0 $PID 2>/dev/null || { echo "server died"; tail -15 $LOG; break; }
done
kill $PID; wait $PID 2>/dev/null
for p in $(pgrep -f "python .*serve_kernels.py new serve .*--name qwen27"); do [ "$p" != "$$" ] && kill $p 2>/dev/null; done
sleep 2; nvidia-smi --query-gpu=memory.used --format=csv,noheader
