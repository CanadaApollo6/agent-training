#!/bin/bash
# On the training pod: runs `specforge train` on a config with the CUDA 13 runtime pieces in place (see
# train_setup.sh), logging to /tmp/train-<name>.log.
#
#   bash train_run.sh smoke-1gpu.yaml
set -uo pipefail
if [ -d /usr/local/cuda-13.0/compat ] && [ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)" -lt 580 ]; then
  export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
fi
[ -d /usr/local/cuda-13.0/bin ] && export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export PATH=$HOME/sf/bin:$PATH
cfg=$(realpath "$1"); name=$(basename "$1" .yaml)
cd ~/SpecForge && specforge train --config "$cfg" > /tmp/train-$name.log 2>&1
echo "TRAIN_EXIT $?" >> /tmp/train-$name.log
