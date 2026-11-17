#!/bin/bash
# On an 8x A100 80GB pod, after setup_qwen27_pod.sh: three vLLM servers for the harness-hub pi adapter test, one arm
# each, tensor parallel 2, 131K context, on the pod's 127.0.0.1:
#   8000 GPUs 0,1  base, reasoning_effort medium, the model's own sampling (T 1.0, top_k 20, top_p 0.95)
#                  names <P>-med-a1..a3
#   8001 GPUs 2,3  base, thinking off, T 0.8 (top_k 20, top_p 0.95 kept)          names <P>-off-a1..a3
#   8002 GPUs 4,5  base + LoRA, thinking off, T 0.8                                names <P>-ad-a1..a3 (the LoRA)
# The two thinking-off arms match the adapter card's eval (thinking off, T 0.8); medium is how we run Qwen3.8-27B.
# One LoRA name per attempt, so --max-loras 3 lets them share a batch. Tool parser qwen3_xml (the card's) for all
# three arms. Waits until all three answer. Logs: /tmp/vllm-<port>.log
#
#   bash serve_qwen27_pod.sh <label prefix, e.g. q27pi>
set -euo pipefail
P=$1
[ -f ~/compat_path ] && export LD_LIBRARY_PATH=$(cat ~/compat_path)
export PATH=$HOME/vllm/bin:$PATH   # ninja: vLLM JIT-builds kernels at startup
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CONFIG_ROOT=$HOME/.vllm
# Prefix caching on this hybrid model needs --mamba-cache-mode align (adapter card); use it if this vLLM has it.
ALIGN=(); vllm serve --help=all 2>/dev/null | grep -q -- --mamba-cache-mode && ALIGN=(--mamba-cache-mode align)
COMMON=(--host 127.0.0.1 --tensor-parallel-size 2 --max-model-len 131072 --gpu-memory-utilization 0.90
  --enable-prefix-caching "${ALIGN[@]}" --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml)
OFF='{"enable_thinking": false}'
T08='{"temperature": 0.8}'
start() { # <gpus> <port> <args...>
  local G=$1 PORT=$2; shift 2
  CUDA_VISIBLE_DEVICES=$G setsid nohup vllm serve ~/qwen27 --port $PORT "${COMMON[@]}" "$@" \
    > /tmp/vllm-$PORT.log 2>&1 < /dev/null &
}
start 0,1 8000 --served-model-name $P-med-a1 $P-med-a2 $P-med-a3 \
  --default-chat-template-kwargs '{"reasoning_effort": "medium"}'
start 2,3 8001 --served-model-name $P-off-a1 $P-off-a2 $P-off-a3 \
  --default-chat-template-kwargs "$OFF" --override-generation-config "$T08"
start 4,5 8002 --served-model-name $P-base-unused --default-chat-template-kwargs "$OFF" \
  --override-generation-config "$T08" --enable-lora --max-lora-rank 32 --max-loras 3 \
  --lora-modules $P-ad-a1=$HOME/pi-lora $P-ad-a2=$HOME/pi-lora $P-ad-a3=$HOME/pi-lora
for PORT in 8000 8001 8002; do
  until curl -sf -m 10 127.0.0.1:$PORT/health > /dev/null; do
    pgrep -f "port $PORT" > /dev/null || { echo "server $PORT died"; tail -40 /tmp/vllm-$PORT.log; exit 1; }
    sleep 15
  done
  echo "up $PORT: $(grep -E 'Maximum concurrency' /tmp/vllm-$PORT.log | tail -1)"
done
echo "align: ${ALIGN[*]:-not supported}"
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '; echo
echo SERVE_DONE
