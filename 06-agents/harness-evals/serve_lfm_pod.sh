#!/bin/bash
# Serve the released M3TiS #3 (CanadaApollo6/LFM2.5-24B-A2B-M3TiS-3090) on a fresh Prime pod with an sm_86 card
# (A10, A40, A6000), one server per conversation lane, ports 8081.. on the pod's 127.0.0.1. Run on the pod:
#
#   N=2 bash serve_lfm_pod.sh        # N servers (a 48 GB card fits 2 at the 131K window)
#
# The release's binary needs a CUDA 13 driver; older drivers (lambdalabs: 570) get NVIDIA's forward-compatibility
# libraries. Logs: /tmp/lfm-<port>.log.
set -euo pipefail
N=${N:-1}
export PATH=$HOME/.local/bin:$PATH
if [ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)" -lt 580 ] && [ ! -d /usr/local/cuda-13.0/compat ]; then
  while pgrep -x apt >/dev/null || pgrep -x apt-get >/dev/null || pgrep -f "[/]usr/bin/unattended-upgrade" >/dev/null; do sleep 5; done
  cd /tmp && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null
  sudo apt-get -o DPkg::Lock::Timeout=900 update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 install -y -qq cuda-compat-13-0 > /tmp/apt.log 2>&1
fi
[ -d /usr/local/cuda-13.0/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
[ -x ~/m3tis-lfm/bin/tensorfold-native ] || {
  uvx --quiet --from huggingface_hub hf download CanadaApollo6/LFM2.5-24B-A2B-M3TiS-3090 --local-dir ~/m3tis-lfm > /tmp/hf.log 2>&1
  chmod +x ~/m3tis-lfm/m3tis-serve.sh ~/m3tis-lfm/bin/tensorfold-native
}
# BIN=<path>: run this server build instead of the released one (the eval's penalty build, fork 6aa76186)
[ -z "${BIN:-}" ] || install -m755 "$BIN" ~/m3tis-lfm/bin/tensorfold-native
for i in $(seq 1 $N); do
  port=$((8080 + i))
  PORT=$port setsid nohup ~/m3tis-lfm/m3tis-serve.sh > /tmp/lfm-$port.log 2>&1 < /dev/null &
  until curl -sf -m 5 127.0.0.1:$port/v1/models > /dev/null; do
    kill -0 $! 2>/dev/null || { echo "server $port died"; tail -5 /tmp/lfm-$port.log; exit 1; }; sleep 2; done
  grep serving /tmp/lfm-$port.log
done
echo SERVE_DONE
