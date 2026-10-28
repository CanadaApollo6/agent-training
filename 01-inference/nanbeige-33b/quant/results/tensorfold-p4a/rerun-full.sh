#!/usr/bin/env bash
set -euo pipefail
cd /home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama
kv_model=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb
kv_receipts=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p4a
python3 tools/zig/llama_kv_prepare.py --review /tmp/nanbeige-review-9a5732e0 --out /tmp/nanbeige-p4a
python3 tools/zig/llama_kv_full.py --model "$kv_model" --prompts /tmp/nanbeige-p4a --receipts "$kv_receipts" --work /tmp/nanbeige-p4a/full --kind "${1:-long}" --sequence "${2:-0}"
