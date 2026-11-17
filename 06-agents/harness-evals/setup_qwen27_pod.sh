#!/bin/bash
# On a Prime pod: vLLM in a uv venv, then Qwen/Qwen3.8-27B (BF16, ~54 GB) and the harness-hub pi LoRA, for the
# adapter test (serve_qwen27_pod.sh). Records the revisions it got. Log: /tmp/setup.log
set -euo pipefail
command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1; }
export PATH=$HOME/.local/bin:$PATH
# vLLM's wheels are CUDA 13; older drivers (< 580) need NVIDIA's forward-compat libcuda (see setup_nanbeige_pod.sh).
if [ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)" -lt 580 ]; then
  $(command -v sudo) env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-13-0 > /tmp/compat.log 2>&1
  echo /usr/local/cuda-13.0/compat > ~/compat_path
fi
# Triton compiles a small C helper against Python.h at startup; a system Python 3.12 without its -dev package fails it
# (datacrunch image, 2026-11-17).
[ -f /usr/include/python3.12/Python.h ] || $(command -v sudo) env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3.12-dev > /tmp/pydev.log 2>&1
uv venv -q ~/vllm --python 3.12
VIRTUAL_ENV=~/vllm uv pip install -q vllm "huggingface_hub[hf_xet]" ninja 2>&1 | tail -3
[ -f ~/compat_path ] && export LD_LIBRARY_PATH=$(cat ~/compat_path)
~/vllm/bin/python -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)'
echo VLLM_DONE
~/vllm/bin/hf download Qwen/Qwen3.8-27B --local-dir ~/qwen27 > /tmp/hf.log 2>&1
~/vllm/bin/hf download harness-hub/Qwen3.8-27B-pi-adapted --local-dir ~/pi-lora >> /tmp/hf.log 2>&1
~/vllm/bin/python - << 'EOF'
from huggingface_hub import HfApi
for r in ["Qwen/Qwen3.8-27B", "harness-hub/Qwen3.8-27B-pi-adapted"]:
    print("rev", r, HfApi().model_info(r).sha)
EOF
du -sh ~/qwen27 ~/pi-lora
echo SETUP_DONE
