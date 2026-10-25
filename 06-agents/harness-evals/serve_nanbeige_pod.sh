#!/bin/bash
# On the pod, after setup_nanbeige_pod.sh: Nanbeige4.3-33B BF16 on vLLM, BF16 KV cache, 262K context, on the pod's
# 127.0.0.1:8000, served as the given names. Parsers from the card (qwen3 reasoning, qwen3_coder tool calls). Its
# generation_config sets no sampling, so the card's agent sampling (T 1.0, top_p 0.95) is set server-side. The
# template keeps earlier turns' thinking by default (preserve_thinking unset), as in their agent evals.
# Waits until it generates. Log: /tmp/vllm.log
#
#   bash serve_nanbeige_pod.sh nanbeige-bf16-a1 [more names]
set -euo pipefail
pkill -f "vllm serve" || true
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat   # see setup_nanbeige_pod.sh
export PATH=$HOME/vllm/bin:$PATH   # ninja: vLLM JIT-builds kernels at startup
# FlashInfer JIT-builds its top-p sampler with the system nvcc, which is CUDA 12.4 here and fails. The PyTorch sampler
# draws from the same distribution.
export VLLM_USE_FLASHINFER_SAMPLER=0
nohup ~/vllm/bin/vllm serve ~/nanbeige --served-model-name "$@" --host 127.0.0.1 --port 8000 \
  --max-model-len 262144 --gpu-memory-utilization 0.92 --enable-prefix-caching \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --override-generation-config '{"temperature": 1.0, "top_p": 0.95}' > /tmp/vllm.log 2>&1 &
until curl -sf -m 10 127.0.0.1:8000/health > /dev/null; do
  pgrep -f "vllm serve" > /dev/null || { tail -40 /tmp/vllm.log; exit 1; }
  sleep 15
done
grep -E "KV cache|Maximum concurrency" /tmp/vllm.log | tail -2
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo SERVE_DONE
