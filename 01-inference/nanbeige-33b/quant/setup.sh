#!/bin/bash
# On a quant pod: exllamav3 1.6.5 (prebuilt wheel, torch 2.10 cu128) in ~/exl3, its convert and eval scripts in ~/qkit
# (outside the source tree, so Python imports the installed wheel), and Nanbeige4.3-33B bf16 at the pinned revision.
# Run from the directory holding model_diff_rows.patch.py. Log lines: SETUP_DONE at the end.
set -euo pipefail
REV=c1093c501cc6460427c0ac634db7223d0450c0ee
EXL3=v1.6.5
WHEEL=https://github.com/turboderp-org/exllamav3/releases/download/$EXL3/exllamav3-1.6.5%2Bcu128.torch2.10.0-cp312-cp312-linux_x86_64.whl
command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1; }
export PATH=$HOME/.local/bin:$PATH
# Nebius H200s ship driver 550 (CUDA 12.4); the forward-compat libcuda runs the cu128 wheels.
if [ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)" -lt 570 ]; then
  if ! apt-cache show cuda-compat-12-8 > /dev/null 2>&1; then
    . /etc/os-release; repo=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID/./}/x86_64
    curl -fsSLo /tmp/cuda-keyring.deb $repo/cuda-keyring_1.1-1_all.deb && sudo dpkg -i /tmp/cuda-keyring.deb > /dev/null
    sudo apt-get update -qq > /dev/null
  fi
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-12-8 > /tmp/compat.log 2>&1 \
    || { echo "cuda-compat-12-8 failed, see /tmp/compat.log"; tail -3 /tmp/compat.log; exit 1; }
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' > ~/compat.sh
else
  : > ~/compat.sh
fi
uv venv -q ~/exl3 --python 3.12
VIRTUAL_ENV=~/exl3 uv pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV=~/exl3 uv pip install -q "$WHEEL" datasets tabulate pyyaml "huggingface_hub[hf_xet]" 2>&1 | tail -3
[ -d ~/exl3src ] || git clone -q https://github.com/turboderp-org/exllamav3.git ~/exl3src
git -C ~/exl3src checkout -q $EXL3
mkdir -p ~/qkit
cp ~/exl3src/convert.py ~/qkit/
~/exl3/bin/python model_diff_rows.patch.py ~/exl3src/eval/model_diff.py ~/qkit/model_diff.py
. ~/compat.sh
~/exl3/bin/python -c 'import torch, importlib.metadata as m, exllamav3; print("torch", torch.__version__, "exllamav3", m.version("exllamav3"), torch.cuda.get_device_name())'
~/exl3/bin/hf download Nanbeige/Nanbeige4.3-33B --revision $REV --local-dir ~/nanbeige > /tmp/hf.log 2>&1
du -sh ~/nanbeige
echo SETUP_DONE
