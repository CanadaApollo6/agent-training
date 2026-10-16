#!/bin/bash
# The reference for the LFM2.5 Terminal-Bench 2 run: Liquid's Q8_0 GGUF on llama-server (llama.cpp a0e6dc7, the
# commit of results/llamacpp-baseline.md), built for the pod's card on a fresh 40 GB+ pod; NP slots of 131072 tokens each,
# fp16 cache, on the pod's 127.0.0.1:8081. Sampling as M3TiS's server does by default, which is Liquid's
# recommendation (temperature 0.1, top_k 50, repetition penalty 1.05 over the last 64 tokens, llama.cpp's own default
# window). Installs NVIDIA's CUDA 12.8 toolkit when CMake can't use the image's (Lambda's images).
# Run on the pod:  NP=6 bash serve_lfm_q8_pod.sh
set -euo pipefail
NP=${NP:-6}
SUDO=; [ "$(id -u)" = 0 ] || SUDO=sudo
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
command -v cmake > /dev/null && command -v git > /dev/null || { $SUDO apt-get update -qq && $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake git build-essential libcurl4-openssl-dev > /tmp/apt.log 2>&1; }
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
mkdir -p ~/models
[ -f ~/models/LFM2.5-24B-A2B-Q8_0.gguf ] || (uvx --quiet --from huggingface_hub hf download LiquidAI/LFM2.5-24B-A2B-GGUF LFM2.5-24B-A2B-Q8_0.gguf --local-dir ~/models > /tmp/hf.log 2>&1) &
DL=$!
[ -x ~/llama.cpp/build/bin/llama-server ] || {
  [ -d ~/llama.cpp ] || git clone -q https://github.com/ggml-org/llama.cpp ~/llama.cpp
  cd ~/llama.cpp && git checkout -q a0e6dc7
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
  configure() { rm -rf build; cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release "$@" > /tmp/cmake.log 2>&1; }
  configure || {  # CMake can't place a distro-packaged toolkit (Lambda's /usr/bin/nvcc): use NVIDIA's
    cd /tmp && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    $SUDO dpkg -i cuda-keyring_1.1-1_all.deb > /dev/null
    $SUDO apt-get -o DPkg::Lock::Timeout=900 update -qq
    $SUDO DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 install -y -qq cuda-nvcc-12-8 \
      cuda-cudart-dev-12-8 libcublas-dev-12-8 cuda-driver-dev-12-8 > /tmp/apt-cuda.log 2>&1
    cd ~/llama.cpp && configure -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
  }
  cmake --build build -j"$(nproc)" --target llama-server > /tmp/build.log 2>&1
}
wait $DL
setsid nohup ~/llama.cpp/build/bin/llama-server -m ~/models/LFM2.5-24B-A2B-Q8_0.gguf -ngl 99 -fa on -np $NP \
  -c $((NP * 131072)) --jinja --temp 0.1 --top-k 50 --top-p 1.0 --min-p 0 --repeat-penalty 1.05 \
  --repeat-last-n 64 --host 127.0.0.1 --port 8081 --metrics > /tmp/llama-server.log 2>&1 < /dev/null &
until curl -sf -m 5 127.0.0.1:8081/v1/models > /dev/null; do kill -0 $! 2>/dev/null || { tail -20 /tmp/llama-server.log; exit 1; }; sleep 3; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo SERVE_DONE
