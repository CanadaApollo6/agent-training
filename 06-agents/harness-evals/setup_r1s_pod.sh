#!/bin/bash
# One-time setup of a fresh Prime pod (Ubuntu 22.04, driver 580) for serve_r1s_pod.sh: the CUDA 13.0 toolkit from
# NVIDIA's apt repo (TensorFold builds its kernels on first use), g++-12 to match the default gcc 12 (without it nvcc
# fails with "cannot execute cc1plus"), and uv. Boot-time apt runs (unattended-upgrades) hold the dpkg lock for a few
# minutes; this waits them out.
set -e
cd /tmp
while pgrep -x apt >/dev/null || pgrep -x apt-get >/dev/null || pgrep -f "[u]nattended-upgr" >/dev/null; do sleep 5; done
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null
sudo apt-get -o DPkg::Lock::Timeout=900 update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 install -y -qq cuda-toolkit-13-0 g++-12 build-essential > /tmp/apt.log 2>&1
command -v uv > /dev/null || [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
echo SETUP_DONE
