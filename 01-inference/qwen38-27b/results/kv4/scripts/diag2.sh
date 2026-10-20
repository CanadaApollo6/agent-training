#!/bin/bash
# diag2.sh: is the 32K flip deterministic and what does the served engine predict there.
set -u
export LD_LIBRARY_PATH=/usr/local/cuda/compat
M=$HOME/.cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c
echo "== probes $(date +%T)"
bash ~/serve.sh probe-bf16 ~/out/probe probe:32833+32943+32944+32945 --parallel 1 --context 40960
bash ~/serve.sh probe-q4 ~/out/probe probe:32833+32943+32944+32945 --kv-cache q4 --parallel 1 --context 40960
echo "== bf16 vs bf16 gate $(date +%T)"
~/tf-kv4/zig-out/bin/tf-qwen35-cuda $M kv-gate ~/fixture-64k.json ~/out/gate64k-bf16.json bf16 2>&1 | grep -v "^depth.*max [0-9.]*$" | tail -6
echo "== q4 gate 64k $(date +%T)"
~/tf-kv4/zig-out/bin/tf-qwen35-cuda $M kv-gate ~/fixture-64k.json ~/out/gate64k-q4.json q4 2>&1 | grep -v "^depth.*max [0-9.]*$" | tail -6
echo "DIAG2 DONE $(date +%T)"
