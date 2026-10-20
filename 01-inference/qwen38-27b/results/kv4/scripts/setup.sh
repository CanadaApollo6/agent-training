#!/bin/bash
# Fresh pod: CUDA 13 toolkit (BUILD=1) or driver compat only, uv, and the checkpoint (DRAFT=1 adds the drafter).
set -e
trap 'echo SETUP_FAILED line $LINENO' ERR
SUDO=; [ "$(id -u)" = 0 ] || SUDO=sudo
cd /tmp
while pgrep -x apt >/dev/null || pgrep -x apt-get >/dev/null || pgrep -f "[/]usr/bin/unattended-upgrade" >/dev/null; do sleep 5; done
command -v wget >/dev/null || { $SUDO apt-get update -qq; $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq wget curl rsync >/dev/null; }
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
$SUDO dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null
$SUDO apt-get -o DPkg::Lock::Timeout=900 update -qq
V=$(apt-cache search --names-only '^cuda-toolkit-13-[0-9]+$' | awk '{print $1}' | sort -V | tail -1 | sed 's/cuda-toolkit-//')
echo "toolkit $V"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
PK="rsync build-essential g++-12"
[ "${BUILD:-0}" = 1 ] && PK="$PK cuda-toolkit-$V"
[ "$DRV" -lt 580 ] && PK="$PK cuda-compat-$V"
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 install -y -qq $PK > /tmp/apt.log 2>&1
command -v uv > /dev/null || [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
uvx -q --from huggingface_hub hf download TensorFold/Qwen3.8-27B-MLX-4bit --revision 22d8d538154e0e5b6f8dcb1fcc74a60a0798104c > /tmp/hf.log 2>&1
[ "${DRAFT:-0}" = 1 ] && uvx -q --from huggingface_hub hf download z-lab/Qwen3.8-27B-DFlash2 --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 >> /tmp/hf.log 2>&1
echo "driver $DRV toolkit $V"
echo SETUP_DONE
