#!/bin/bash
set -u
A=~/Projects/Personal/Research/agent-training/01-inference; Z=$A/qwen38-27b/zig; TF=$A/tools/TensorFold-moe
export TENSORFOLD_NO_UPDATE_CHECK=1
up() { for i in $(seq 1 240); do curl -sf http://127.0.0.1:$1/v1/models > /dev/null && return 0; sleep 2; done; return 1; }
stop() { for p in $(ps -C tensorfold-nati -o pid=); do kill $p; done; for i in $(seq 30); do ps -C tensorfold-nati -o pid= | grep -q . || break; sleep 1; done; }
cd $TF
for g in 0 1 0 1; do
  TF_QWEN27_GRAPHS=$g setsid ./zig-out/native/bin/tensorfold-native serve ~/models/ornith/r1s --context 65536 --parallel 1 --port 8095 --backend cuda --no-update-check > ~/.cache/r1s-h2h/zig-ab.log 2>&1 < /dev/null &
  up 8095 || exit 1
  echo "== graphs=$g $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)C"; python3 $Z/long_check.py http://127.0.0.1:8095 40000 $TF/ | grep 'draft on'
  stop
done
