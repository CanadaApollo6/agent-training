#!/bin/bash
# On a multi-GPU pod, after setup_nanbeige_pod.sh: one vLLM server a GPU for bulk drafter-data generation, GPU i on
# port 8000+i. FP8 weights and FP8 KV: the weights drop from 66 to 33 GB and each token's KV from 213 to 107 KB, so an
# H200 holds ~3x the parallel requests of the BF16 setup (the pilot's KV-bound ~3K tok/s). SpecForge's own DFlash2
# recipe captures with FP8 targets too. Same sampling and parsers as serve_nanbeige_pod.sh. Logs: /tmp/vllm-<i>.log
#
#   bash serve_gen.sh 0 1 2      # GPUs to serve on
set -euo pipefail
[ -d /usr/local/cuda-13.0/compat ] && [ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)" -lt 580 ] \
  && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat   # see setup_nanbeige_pod.sh
export PATH=$HOME/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
for g in "$@"; do
  CUDA_VISIBLE_DEVICES=$g nohup ~/vllm/bin/vllm serve ~/nanbeige --served-model-name nanbeige --host 127.0.0.1 \
    --port $((8000 + g)) --max-model-len 16384 --gpu-memory-utilization 0.92 --enable-prefix-caching \
    --quantization fp8 --kv-cache-dtype fp8 --max-num-seqs 256 --reasoning-parser qwen3 \
    --override-generation-config '{"temperature": 1.0, "top_p": 0.95}' > /tmp/vllm-$g.log 2>&1 &
  sleep 20   # staggered: parallel cold starts race on the torch.compile cache
done
for g in "$@"; do
  until curl -sf -m 10 127.0.0.1:$((8000 + g))/health > /dev/null; do
    grep -q "Error\|Traceback" /tmp/vllm-$g.log && { tail -30 /tmp/vllm-$g.log; exit 1; }
    sleep 15
  done
  grep -E "KV cache|Maximum concurrency" /tmp/vllm-$g.log | tail -2
done
echo SERVE_DONE
