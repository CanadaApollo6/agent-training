#!/bin/bash
# rewrite.py with bf16 Ornith 1.5 35B-A3B in-process, ENGINES copies of TP GPUs each on shards of the samples
# (8xA100-80GB: ENGINES=4 TP=2; one H200: ENGINES=1 TP=1).
# Expects ~/rewrite.py and ~/teacher.jsonl copied up; writes ~/out/teacher_rewritten.jsonl and ~/out/rewrite.jsonl
# per shard, out/*.<shard>.jsonl (resumable: rerun to continue). Logs in /tmp.  Extra rewrite.py flags pass through:
#   ENGINES=1 TP=2 bash rewrite_pod.sh --limit 4
set -uo pipefail
SUDO=$([ "$(id -u)" = 0 ] || echo sudo)   # runpod containers run as root without sudo
export PATH=$HOME/.local/bin:$PATH
if [ ! -x ~/vllm/bin/python ]; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
  ( uv venv -q ~/vllm --python 3.12 && VIRTUAL_ENV=~/vllm uv pip install -q "vllm==0.30.0" "huggingface_hub[hf_xet]" hf_transfer ) > /tmp/pip.log 2>&1 &
  ( while pgrep -x apt >/dev/null || pgrep -x apt-get >/dev/null || pgrep -f "[/]usr/bin/unattended-upgrade" >/dev/null; do sleep 5; done
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && $SUDO dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null
    $SUDO apt-get -o DPkg::Lock::Timeout=900 update -qq && $SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 install -y -qq cuda-compat-13-0 ) > /tmp/apt.log 2>&1 &
  wait
fi
[ -f ~/ornith-bf16/config.json ] || HF_HUB_ENABLE_HF_TRANSFER=1 ~/vllm/bin/hf download ornith-ai/Ornith-1.5-35B-A3B --local-dir ~/ornith-bf16 > /tmp/download.log 2>&1
echo "$(date +%T) READY" >> /tmp/setup.log
[ -d /usr/local/cuda-13.0/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
ENGINES=${ENGINES:-1} TP=${TP:-1}
cd ~ && for i in $(seq 0 $((ENGINES - 1))); do
  ( CUDA_VISIBLE_DEVICES=$(seq -s, $((i * TP)) $((i * TP + TP - 1))) VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CONFIG_ROOT=$HOME/.vllm \
      ~/vllm/bin/python rewrite.py --samples teacher.jsonl --model ~/ornith-bf16 --shard $i/$ENGINES --tp $TP \
      --out out/teacher_rewritten.$i.jsonl --stats out/rewrite.$i.jsonl "$@" >> /tmp/rewrite.$i.log 2>&1
    echo "$(date +%T) DONE shard $i/$ENGINES exit=$?" >> /tmp/setup.log ) &
done
wait
