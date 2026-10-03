#!/bin/bash
# H100 pod: vLLM 0.30 serving bf16 Ornith 1.5 35B-A3B with prompt log-probs on 127.0.0.1:8200. Logs in /tmp.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
( uv venv -q ~/vllm --python 3.12 && VIRTUAL_ENV=~/vllm uv pip install -q "vllm==0.30.0" "huggingface_hub[hf_xet]" hf_transfer ) > /tmp/pip.log 2>&1 &
( while pgrep -x apt >/dev/null || pgrep -x apt-get >/dev/null || pgrep -f "[/]usr/bin/unattended-upgrade" >/dev/null; do sleep 5; done
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && sudo dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null
  sudo apt-get -o DPkg::Lock::Timeout=900 update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 install -y -qq cuda-compat-13-0 ) > /tmp/apt.log 2>&1 &
wait
HF_HUB_ENABLE_HF_TRANSFER=1 ~/vllm/bin/hf download ornith-ai/Ornith-1.5-35B-A3B --local-dir ~/ornith-bf16 > /tmp/download.log 2>&1
echo DOWNLOADED >> /tmp/setup.log
[ -d /usr/local/cuda-13.0/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CONFIG_ROOT=$HOME/.vllm ~/vllm/bin/vllm serve ~/ornith-bf16 --served-model-name ornith \
  --host 127.0.0.1 --port 8200 --max-model-len 131072 --gpu-memory-utilization 0.95 --max-num-batched-tokens 4096 \
  --max-num-seqs 4 --max-logprobs 1 --language-model-only --trust-remote-code > /tmp/vllm.log 2>&1
