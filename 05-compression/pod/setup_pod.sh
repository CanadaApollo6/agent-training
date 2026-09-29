#!/bin/bash
# Turn a fresh Prime pod (ubuntu_22_cuda_12 image) into a GGUF server: copy PrismML's prebuilt llama.cpp, install its
# CUDA 12.8 runtime from pip, install uv, write ~/serve.sh, and download the given HF files into ~/models.
#
#   05-compression/pod/setup_pod.sh <ip> [repo:file ...]
#   e.g. setup_pod.sh 1.2.3.4 ornith-ai/Ornith-1.5-9B-GGUF:Ornith-1.5-9B-Q8_0.gguf
#
# On the pod: ~/serve.sh <gguf> [slots] [ctx per slot] [port] serves it on 127.0.0.1:<port> (default 8080). Reach it
# with an SSH tunnel: ssh -f -N -L <local port>:127.0.0.1:<port> ubuntu@<ip>
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IP=$1; shift
SSH="ssh -i $HOME/.ssh/prime_ed25519 -o StrictHostKeyChecking=accept-new ubuntu@$IP"
rsync -az -e "ssh -i $HOME/.ssh/prime_ed25519" "$ROOT/01-inference/tools/prism-llama" "ubuntu@$IP:~/"
$SSH 'bash -s' <<'EOF'
set -e
# the installer exits non-zero on this image (it can't write a fish config) after installing uv fine
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
export PATH=$HOME/.local/bin:$PATH
uv --version >/dev/null
uv venv -q --allow-existing ~/cu --python 3.12
VIRTUAL_ENV=~/cu uv pip install -q "nvidia-cuda-runtime-cu12==12.8.*" "nvidia-cublas-cu12==12.8.*" \
  "nvidia-cuda-nvrtc-cu12==12.8.*"
cat > ~/serve.sh <<"SERVE"
#!/bin/bash
# serve.sh <gguf> [slots] [ctx per slot] [port]: llama-server, 8-bit KV, Ornith/Qwen card sampling defaults
M=$1; NP=${2:-8}; CTX=${3:-65536}; PORT=${4:-8080}
LIBS=$(ls -d ~/cu/lib/python3.12/site-packages/nvidia/*/lib | tr "\n" ":")
pkill -f "llama-server.*--port $PORT" || true; sleep 2
LD_LIBRARY_PATH=$LIBS:$HOME/prism-llama nohup ~/prism-llama/llama-server -m "$M" -ngl 99 -fa on -np $NP \
  -c $((NP*CTX)) -ctk q8_0 -ctv q8_0 --jinja --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
  --host 127.0.0.1 --port $PORT --metrics > ~/server-$PORT.log 2>&1 &
until curl -sf 127.0.0.1:$PORT/health >/dev/null; do
  sleep 3; pgrep -f "llama-server.*--port $PORT" >/dev/null || { tail -20 ~/server-$PORT.log; exit 1; }
done
echo "up on $PORT: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
SERVE
chmod +x ~/serve.sh
mkdir -p ~/models
EOF
for spec in "$@"; do
  $SSH "cd ~/models && ~/.local/bin/uvx --from huggingface_hub hf download ${spec%%:*} ${spec#*:} --local-dir . \
    >> ~/dl.log 2>&1 && echo 'got ${spec#*:}'"
done
