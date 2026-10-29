#!/usr/bin/env bash
set -euo pipefail
p3_engine=${TF_ENGINE:-/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama}
p3_model=${TF_LLAMA_MODEL:-/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/quant/models/3.5-nb}
p3_draft=${TF_LLAMA_DRAFT:-/home/riels/Projects/Personal/Research/agent-training/01-inference/nanbeige-33b/drafter/data/draft-cool-step570}
p3_receipts=$(cd -- "$(dirname -- "$0")" && pwd)
p3_kind=${1:-short}
p3_kv=${2:-16}
p3_format=${3:-affine4}
case "$p3_kind" in short) p3_timeout=3600 ;; long) p3_timeout=14400 ;; *) exit 2 ;; esac
cd "$p3_engine"
python3 tools/zig/llama_kv_prepare.py --review "${P3_REVIEW:-/tmp/nanbeige-review-9a5732e0}" --out /tmp/nanbeige-p3/full-inputs --length "${P3_LENGTH:-20000}"
python3 tools/zig/llama_draft_full.py --model "$p3_model" --draft "$p3_draft" \
  --prompts "/tmp/nanbeige-p3/full-inputs/$p3_kind/sequences.json" \
  --out "$p3_receipts/full-$p3_kind-kv$p3_kv-$p3_format" --format "$p3_format" --kv "$p3_kv" \
  --tokens 256 --reps 5 --seeds "${P3_SEEDS:-32}" --timeout "${P3_TIMEOUT:-$p3_timeout}"
