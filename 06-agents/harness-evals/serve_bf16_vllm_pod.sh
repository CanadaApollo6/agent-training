#!/bin/bash
# Serve Ornith 1.5 35B-A3B in bf16 on a 2-GPU Prime pod (2x A6000, 96 GB) with vLLM, as the model card's recipe does
# (tensor parallel 2, 256K context, prefix caching, the qwen3_xml tool parser and qwen3 reasoning parser), with
# messages_proxy.py in front on 127.0.0.1:8080 for Claude Code: the same Messages translation TensorFold's endpoint uses
# (anthropic.py, copied to ~/anthropic.py). Installs uv and vLLM, downloads the checkpoint from Hugging Face.
# Logs: /tmp/vllm.log, /tmp/proxy.log.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
[ -d ~/vllm ] || uv venv -q ~/vllm --python 3.12
VIRTUAL_ENV=~/vllm uv pip install -q "vllm>=0.19.1" "huggingface_hub[hf_xet]"
[ -f ~/models/ornith-bf16/config.json ] || ~/vllm/bin/hf download ornith-ai/Ornith-1.5-35B-A3B --local-dir ~/models/ornith-bf16 > /tmp/download.log 2>&1
# vLLM's sampler without FlashInfer, whose kernels compile on first use and need nvcc (these images have none); its
# config in ~/.vllm, since ~/.config belongs to root here
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CONFIG_ROOT=$HOME/.vllm setsid nohup ~/vllm/bin/vllm serve ~/models/ornith-bf16 --served-model-name Ornith-1.5-35B-A3B \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 2 --max-model-len 262144 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --trust-remote-code > /tmp/vllm.log 2>&1 < /dev/null &
for i in $(seq 360); do curl -sf -m 2 127.0.0.1:8000/v1/models >/dev/null && break; sleep 5; done
curl -sf -m 2 127.0.0.1:8000/v1/models >/dev/null || { echo "vLLM didn't come up"; tail -30 /tmp/vllm.log; exit 1; }
setsid nohup python3 ~/messages_proxy.py --upstream http://127.0.0.1:8000 --port 8080 --context 262144 \
  --translator ~/anthropic.py > /tmp/proxy.log 2>&1 < /dev/null &
for i in $(seq 30); do curl -sf -m 2 127.0.0.1:8080/v1/models >/dev/null && { echo up; exit 0; }; sleep 2; done
echo "proxy didn't come up"; cat /tmp/proxy.log; exit 1
