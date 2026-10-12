#!/usr/bin/env bash
# The 3090's Triton glue set for TensorFold PR #608's Zig 27B engine, recorded from Python 0.6.6 (tf066-sm86-oracle.patch),
# then the Zig engine's serial and drafted replies checked against 0.6.6's. Work dir ~/.cache/tf066; about 5 minutes of GPU.
set -euo pipefail
R=$(cd "$(dirname "$0")/../../.." && pwd)                 # agent-training
Z=$R/01-inference/tools/TensorFold-zig                     # pr-608 + pr608-sm86-gemv-rows.patch, built with -Dsm=86
W=~/.cache/tf066; mkdir -p $W/tmp $W/out
M=$(ls -d ~/.cache/huggingface/hub/models--Vontra--Qwen3.8-27B-MLX-4bit/snapshots/*)
D=$(ls -d ~/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/*)
cd $R/01-inference/envs/tf066
run() { TMPDIR=$W/tmp TRITON_CACHE_DIR=$W/triton-$1 TORCH_EXTENSIONS_DIR=$W/torch_ext uv run python -B "${@:2}"; }
run serial $Z/zig/tests/cuda/qwen27/capture.py --model $M --out $W/out/serial --tools $Z/tools/zig
run draft $Z/zig/tests/cuda/qwen27/capture_draft.py --model $M --draft $D --prompts $W/out/serial/prompts.json --out $W/out/draft --tools $Z/tools/zig
uv run python -B $R/01-inference/qwen38-27b/zig/jit_all.py $W/out/jit-all.json $Z/zig/tests/cuda/qwen27   # capture.py's jit.json lacks the drafter's
cd $Z
for c in serial draft; do python3 -B tools/zig/triton_aot_manifest.py --cache $W/triton-$c --launches $W/out/$c/launches.json --out $W/out/$c/manifest.json; done
python3 -B zig/tests/cuda/nemotron/aot_pack.py --manifest $W/out/serial/manifest.json --cache $W/triton-serial \
  --manifest $W/out/draft/manifest.json --cache $W/triton-draft --jit $W/out/jit-all.json --out $W/aot-sm86
./zig-out/bin/tf-qwen27-cuda run $M --kernels $W/aot-sm86 --prompts $W/out/serial/prompts.json --expect $W/out/serial/results.json
./zig-out/bin/tf-qwen27-cuda draft $M --kernels $W/aot-sm86 --drafter $D --prompts $W/out/serial/prompts.json --expect $W/out/draft/draft.json
