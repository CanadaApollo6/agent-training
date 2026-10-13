#!/bin/bash
# Zig server with TF_QWEN27_GRAPHS=1: conc.py at --parallel 4, then the h2h bench at --parallel 1.
set -u
A=~/Projects/Personal/Research/agent-training/01-inference
H=$A/speed-hillclimb/tensorfold; Z=$A/qwen38-27b/zig; OUT=$Z/results/r1s; TF=$A/tools/TensorFold-moe
export TENSORFOLD_NO_UPDATE_CHECK=1 TF_QWEN27_GRAPHS=1
up() { for i in $(seq 1 240); do curl -sf http://127.0.0.1:$1/v1/models > /dev/null && return 0; sleep 2; done; return 1; }
stop() { for p in $(ps -C tensorfold-nati -o pid=); do kill $p; done; for i in $(seq 30); do ps -C tensorfold-nati -o pid= | grep -q . || break; sleep 1; done; }
cd $TF
setsid ./zig-out/native/bin/tensorfold-native serve ~/models/ornith/r1s --context 32768 --parallel 4 --port 8095 --backend cuda --no-update-check > ~/.cache/r1s-h2h/zig-g4.log 2>&1 < /dev/null &
up 8095 || { tail -20 ~/.cache/r1s-h2h/zig-g4.log; exit 1; }
echo "== conc (parallel 4)"; python3 $Z/conc.py http://127.0.0.1:8095
stop
setsid ./zig-out/native/bin/tensorfold-native serve ~/models/ornith/r1s --context 65536 --parallel 1 --port 8095 --backend cuda --no-update-check > ~/.cache/r1s-h2h/zig-g1.log 2>&1 < /dev/null &
up 8095 || { tail -20 ~/.cache/r1s-h2h/zig-g1.log; exit 1; }
echo "== zig-graphs"; nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader
(cd $A/envs/tensorfold060 && uv run python $A/tools/TensorFold-060/tools/bench_openai.py http://127.0.0.1:8095 r1s --tokens 256 --reps 3 --temperatures 1.0,0 --label zig-graphs --output $OUT/h2h-bench-zig-graphs.json 2>&1 | tail -5)
for n in 16000 40000; do MODEL_NAME=r1s python3 $H/longctx.py 8095 $n; done
python3 $Z/long_check.py http://127.0.0.1:8095 40000 $TF/
stop
