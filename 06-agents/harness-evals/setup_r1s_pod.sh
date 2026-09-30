#!/bin/bash
# One-time setup of a fresh Prime pod (Ubuntu 22.04, driver 580) for serve_r1s_pod.sh: the CUDA 13.0 toolkit from
# NVIDIA's apt repo (TensorFold builds its kernels on first use), g++-12 to match the default gcc 12 (without it nvcc
# fails with "cannot execute cc1plus"), and uv. The apt lock may be held by unattended-upgrades for a minute after boot.
set -e
cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-13-0 g++-12 build-essential > /tmp/apt.log 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
echo SETUP_DONE
