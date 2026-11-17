#!/bin/bash
# On a Prime pod (4x A6000, sm_86): a uv venv for the build and KL check (torch, transformers, fla, mlx CPU), then the
# five repos: Qwen BF16, the harness-hub LoRA (both at the revisions the pilot used), Vontra's MLX 4-bit weights,
# the DFlash2 drafter and the M3TiS #1 release bundle. Records what it got in ~/out/revisions.txt.
# Log: /tmp/setup.log
set -euo pipefail
command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1; }
export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/out
# The release binary needs a CUDA 13 driver (R580+); older drivers get NVIDIA's forward-compat libcuda.
if [ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)" -lt 580 ]; then
  $(command -v sudo) env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-13-0 > /tmp/compat.log 2>&1
  echo /usr/local/cuda-13.0/compat > ~/compat_path
fi
[ -f /usr/include/python3.12/Python.h ] || $(command -v sudo) env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3.12-dev > /tmp/pydev.log 2>&1 || true
uv venv -q ~/kl --python 3.12
VIRTUAL_ENV=~/kl uv pip install -q torch transformers accelerate safetensors numpy "mlx[cpu]" flash-linear-attention \
  "huggingface_hub[hf_xet]" 2>&1 | tail -3
~/kl/bin/python -c 'import torch, transformers, mlx.core; print("torch", torch.__version__, "cuda", torch.version.cuda,
"transformers", transformers.__version__, "mlx", mlx.core.__version__)' | tee ~/out/versions.txt
echo VENV_DONE
dl() { ~/kl/bin/hf download "$1" ${3:+--revision $3} --local-dir "$2" >> /tmp/hf.log 2>&1; }
dl Vontra/Qwen3.8-27B-MLX-4bit ~/vontra &
dl z-lab/Qwen3.8-27B-DFlash2 ~/drafter &
dl CanadaApollo6/Qwen3.8-27B-M3TiS-3090 ~/m3tis &
dl harness-hub/Qwen3.8-27B-pi-adapted ~/pi-lora e786f8491742268f9b2bc729697b13bfac26d4ad &
dl Qwen/Qwen3.8-27B ~/qwen27 37561816e66b5b9153b1b64c10968c18c1e6f4a0 &
wait
chmod +x ~/m3tis/m3tis-serve.sh ~/m3tis/bin/*
~/kl/bin/python - << 'EOF' | tee ~/out/revisions.txt
from huggingface_hub import HfApi
for r in ["Qwen/Qwen3.8-27B", "harness-hub/Qwen3.8-27B-pi-adapted", "Vontra/Qwen3.8-27B-MLX-4bit",
          "z-lab/Qwen3.8-27B-DFlash2", "CanadaApollo6/Qwen3.8-27B-M3TiS-3090"]:
    print(r, HfApi().model_info(r).sha)
EOF
du -sh ~/qwen27 ~/pi-lora ~/vontra ~/drafter ~/m3tis
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee ~/out/gpus.txt
echo SETUP_DONE
