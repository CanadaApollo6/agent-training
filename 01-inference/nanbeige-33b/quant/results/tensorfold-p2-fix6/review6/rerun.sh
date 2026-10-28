#!/usr/bin/env bash
set -euo pipefail
review_dir=/tmp/nanbeige-review-891015b9
engine_dir=/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama
export CUDA_VISIBLE_DEVICES=''
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$review_dir/deps"
export TF_LLAMA_MODEL=/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb
cd "$engine_dir"
python3 "$review_dir/build_routes.py"
"$HOME/exl3-local/bin/python" "$review_dir/independent.py" generate > "$review_dir/generate.log" 2>&1
REVIEW_INPUT="$review_dir/input.jsonl" "$review_dir/routes" > "$review_dir/routes.log" 2>&1
"$HOME/exl3-local/bin/python" "$review_dir/independent.py" compare
"$HOME/exl3-local/bin/python" "$review_dir/targeted.py" > "$review_dir/targeted-generate.log" 2>&1
REVIEW_INPUT="$review_dir/targeted/input.jsonl" "$review_dir/routes" > "$review_dir/targeted/routes.log" 2>&1
"$HOME/exl3-local/bin/python" -c "import sys; from pathlib import Path; sys.path.insert(0, '$review_dir'); import independent; independent.ROOT=Path('$review_dir/targeted'); independent.compare()"
"$HOME/.local/share/zig-x86_64-linux-0.17.0/zig" build test-llama-server-host test-server-cpu test-tool-parse-host test-server-recovery-host test-tool-visibility-http-host -j4 --summary all --cache-dir "$review_dir/cache" > "$review_dir/tests-fresh.log" 2>&1
python3 tools/zig/lean_check.py > "$review_dir/lean-head.txt"
python3 tools/zig/lean_new.py /home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/results/tensorfold-p2-fix1/lean-base.txt "$review_dir/lean-head.txt"
