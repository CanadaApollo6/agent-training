#!/usr/bin/env bash
set -euo pipefail
cd /home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama
TF_MODEL=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb
TF_RESULTS=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p1
TF_FIX="$TF_RESULTS/fix1"
python3 tools/zig/llama_gpu_run.py --need-mib 17000 --timeout 180 --receipt "$TF_FIX/full52-logits-vram.json" -- "$TF_FIX/tf-llama-after" llama-logits "$TF_MODEL" "$TF_RESULTS/dense" "$TF_FIX/full52"
python3 tools/zig/llama_gpu_run.py --need-mib 17000 --timeout 180 --receipt "$TF_FIX/full52-greedy-vram.json" -- "$TF_FIX/tf-llama-after" llama-greedy "$TF_MODEL" "$TF_RESULTS/dense" 128 "$TF_FIX/full52-greedy.json"
python3 tools/zig/llama_fix_bench.py "$TF_FIX/tf-llama-before" "$TF_FIX/tf-llama-after" "$TF_MODEL" "$TF_RESULTS" "$TF_FIX/bench"
