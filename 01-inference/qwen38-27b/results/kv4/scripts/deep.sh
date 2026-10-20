#!/bin/bash
# The deep gate: wait for setup, build, then kv-gate over the full fixture for each format given (default q4).
set -u
until grep -q "SETUP_DONE\|SETUP_FAILED" /tmp/setup.log; do sleep 10; done
grep -q SETUP_FAILED /tmp/setup.log && { echo "setup failed"; exit 1; }
BASE=0 TESTS=0 bash ~/build.sh > ~/build-all.log 2>&1
grep -q "kv4 build EXIT 0" ~/build-kv4.log || { echo "build failed"; exit 1; }
M=$HOME/.cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c
mkdir -p ~/out
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > ~/out/gpu.txt
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > ~/out/smi-gate.txt &
MON=$!
for f in ${FORMATS:-q4}; do
  ( time ~/tf-kv4/zig-out/bin/tf-qwen35-cuda $M kv-gate ~/${FIXTURE:-fixture-full.json} ~/out/gate-$f.json $f ) > ~/out/gate-$f.log 2>&1
  echo "gate $f EXIT $?" >> ~/out/gate-$f.log
done
kill $MON
echo DEEP_DONE
