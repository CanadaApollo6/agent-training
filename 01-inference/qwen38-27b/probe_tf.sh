#!/bin/bash
# usage: probe_tf.sh LABEL [extra tensorfold serve args]
# The 20-problem math probe (reasoning_probe.py, server backend) against TensorFold serving Qwen3.8-27B on the 3090,
# with the same guards as bench_tf.sh. PROBE_ARGS: extra reasoning_probe.py flags (e.g. --no-draft). Env (TENSORFOLD_KV_BITS etc.) passes through to the server.
set -u
LABEL=$1; shift
cd ~/Projects/Personal/Research/agent-training/01-inference/envs/tensorfold
export TENSORFOLD_CUDA_RESERVE_GIB=${TENSORFOLD_CUDA_RESERVE_GIB:-1} CUDA_HOME=/opt/cuda PATH=/opt/cuda/bin:$PATH TENSORFOLD_NO_UPDATE_CHECK=1
H=~/Projects/Personal/Research/agent-training/01-inference
LOG=$H/qwen38-27b/results/serve-$LABEL.log
rm -f ~/.cache/torch_extensions/py312_cu130/tensorfold_{gdn_v2,qmm_v4}/lock
TF_GPU_ONLY_BUDGET=1 systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=0 uv run python $H/speed-hillclimb/tensorfold/serve_kernels.py new serve ${MODEL:-Vontra/Qwen3.8-27B-MLX-4bit} --name local --port 8092 --prompt-cache-gib 0 "$@" > $LOG 2>&1 &
PID=$!
for i in $(seq 1 300); do
  curl -sf http://127.0.0.1:8092/v1/models > /dev/null && break
  kill -0 $PID 2>/dev/null || { echo "server died"; tail -30 $LOG; exit 1; }
  sleep 2
done
echo "up after $((i*2)) s"
cd $H/qwen38-27b
uv run --project $H/envs/prism-llama python reasoning_probe.py server --url http://127.0.0.1:8092/v1 --label $LABEL \
  --deadline-min ${DEADLINE_MIN:-25} ${PROBE_ARGS:-} 2>&1 | grep -v "^ *problem\|^ *\.\.\." | tail -30
kill $PID; wait $PID 2>/dev/null
for p in $(pgrep -f "python .*serve_kernels.py new serve .*--port 8092"); do [ "$p" != "$$" ] && kill $p 2>/dev/null; done
sleep 2; nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader
