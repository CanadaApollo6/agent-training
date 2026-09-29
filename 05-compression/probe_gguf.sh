#!/bin/bash
# Run the long-reasoning probe (01-inference/qwen38-27b/reasoning_probe.py) on any GGUF model:
# start PrismML's llama-server (it runs standard GGUF types too), probe it with 8 requests at once, stop the server.
#
#   05-compression/probe_gguf.sh <path/to/model.gguf> <label> [results dir] [extra reasoning_probe.py args...]
#
# Needs 01-inference/tools/prism-llama (the prebuilt CUDA 12.8 release) and envs/prism-llama (its CUDA 12 libraries).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=$1; LABEL=$2; OUT=${3:-$ROOT/05-compression/ornith/results}; shift $(( $# < 3 ? $# : 3 ))
LIBS=$(ls -d "$ROOT"/01-inference/envs/prism-llama/.venv/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')
# 8 slots x (16K answer + prompt), 8-bit KV cache: the same settings as the EXL3 and Bonsai runs
LD_LIBRARY_PATH=$LIBS:$ROOT/01-inference/tools/prism-llama "$ROOT"/01-inference/tools/prism-llama/llama-server \
  -m "$MODEL" -ngl 99 -fa on -c 139264 -np 8 -ctk q8_0 -ctv q8_0 --jinja --host 127.0.0.1 --port 8080 \
  > "/tmp/llama-server-$LABEL.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
until curl -sf http://127.0.0.1:8080/health >/dev/null; do
  sleep 3; kill -0 $SERVER 2>/dev/null || { echo "server died:"; tail -20 "/tmp/llama-server-$LABEL.log"; exit 1; }
done
cd "$ROOT" && PYTHONUNBUFFERED=1 uv run --project 01-inference/envs/prism-llama \
  01-inference/qwen38-27b/reasoning_probe.py server --label "$LABEL" --out "$OUT" "$@"
