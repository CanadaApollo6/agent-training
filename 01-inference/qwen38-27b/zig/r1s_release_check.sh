#!/bin/bash
# M3TiS #2 release check: the staged bundle's own launcher at its default 262K window, one stream.
set -u
A=~/Projects/Personal/Research/agent-training/01-inference
H=$A/speed-hillclimb/tensorfold; Z=$A/qwen38-27b/zig; OUT=$Z/results/r1s; TF=$A/tools/TensorFold-moe
B=${BUNDLE:-~/.cache/m3tis2/bundle}; LOG=~/.cache/r1s-h2h/release.log
up() { for i in $(seq 1 240); do curl -sf http://127.0.0.1:$1/v1/models > /dev/null && return 0; sleep 2; done; return 1; }
stop() { for p in $(ps -C tensorfold-nati -o pid=); do kill $p; done; for i in $(seq 30); do ps -C tensorfold-nati -o pid= | grep -q . || break; sleep 1; done; }
PORT=8095 setsid $B/m3tis-serve.sh > $LOG 2>&1 < /dev/null &
up 8095 || { tail -20 $LOG; exit 1; }
echo "== loaded"; nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv,noheader
curl -s http://127.0.0.1:8095/v1/models; echo
echo "== conc (same replies)"; python3 $Z/conc.py http://127.0.0.1:8095
echo "== bench"; (cd $A/envs/tensorfold060 && uv run python $A/tools/TensorFold-060/tools/bench_openai.py http://127.0.0.1:8095 r1s --tokens 256 --reps 3 --temperatures 1.0,0 --label release --output $OUT/release-bench.json 2>&1 | tail -5)
for n in 16000 40000 150000; do MODEL_NAME=r1s python3 $H/longctx.py 8095 $n; nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv,noheader; done
python3 $Z/long_check.py http://127.0.0.1:8095 40000 $TF/
stop
