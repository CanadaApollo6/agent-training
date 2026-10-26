#!/bin/bash
# On a training pod: SpecForge (rev fddcb4a) with its SGLang 0.5.18 capture patch, Mooncake, our Nanbeige chat format,
# and Nanbeige4.3-33B at the pinned revision. Log: /tmp/train_setup.log. Run from the directory holding this script's
# siblings (register_template.py, nanbeige4.3-33b-dflash2.json, allow_shared_gpu.py).
set -euo pipefail
REV=c1093c501cc6460427c0ac634db7223d0450c0ee
command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1; }
export PATH=$HOME/.local/bin:$PATH
# torch 2.13 is CUDA 13; Nebius H200s ship driver 550. The forward-compat libcuda bridges it, and FlashInfer's JIT
# kernels need a CUDA 13 nvcc, not the system 12.4.
if [ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)" -lt 580 ]; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-13-0 cuda-nvcc-13-0 cuda-cudart-dev-13-0 \
    > /tmp/compat.log 2>&1
fi
[ -d ~/SpecForge ] || git clone -q https://github.com/sgl-project/SpecForge.git ~/SpecForge
cd ~/SpecForge && git checkout -q fddcb4a && cd -
uv venv -q ~/sf --python 3.12
VIRTUAL_ENV=~/sf uv pip install -q -e "$HOME/SpecForge[data]" mooncake-transfer-engine-cuda13==0.3.13.post1 ninja \
  "huggingface_hub[hf_xet]" 2>&1 | tail -3
PATH=~/sf/bin:$PATH bash ~/SpecForge/scripts/apply_sglang_spec_capture_patch.sh --target v0.5.18 2>&1 | tail -2
~/sf/bin/python register_template.py
~/sf/bin/python allow_shared_gpu.py
cp nanbeige4.3-33b-dflash2.json ~/SpecForge/configs/
~/sf/bin/python -c 'import torch, sglang; print("torch", torch.__version__, "sglang", sglang.__version__)'
echo SF_DONE
~/sf/bin/hf download Nanbeige/Nanbeige4.3-33B --revision $REV --local-dir ~/nanbeige > /tmp/hf.log 2>&1
du -sh ~/nanbeige
echo SETUP_DONE
