#!/bin/bash
# On a pod: vLLM in a uv venv, then Nanbeige4.3-33B (BF16, 66 GB) at a pinned revision. Log: /tmp/setup.log
set -euo pipefail
REV=c1093c501cc6460427c0ac634db7223d0450c0ee
command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1; }
export PATH=$HOME/.local/bin:$PATH
# vLLM's wheels are CUDA 13; Nebius H200s ship driver 550 (CUDA 12.4). NVIDIA's forward-compat libcuda bridges it.
# Newer drivers (580+) don't need it, and an older compat libcuda would shadow them.
if [ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)" -lt 580 ]; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-13-0 > /tmp/compat.log 2>&1
  export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
fi
uv venv -q ~/vllm --python 3.12
VIRTUAL_ENV=~/vllm uv pip install -q vllm "huggingface_hub[hf_xet]" ninja 2>&1 | tail -3
~/vllm/bin/python -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)'
echo VLLM_DONE
~/vllm/bin/hf download Nanbeige/Nanbeige4.3-33B --revision $REV --local-dir ~/nanbeige > /tmp/hf.log 2>&1
du -sh ~/nanbeige
echo SETUP_DONE
