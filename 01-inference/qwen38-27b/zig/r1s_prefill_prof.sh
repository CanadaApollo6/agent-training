#!/bin/bash
# nsys over one 21K prefill (longctx.py 16000): $1 = zig | python
set -u
A=~/Projects/Personal/Research/agent-training/01-inference; H=$A/speed-hillclimb/tensorfold; TF=$A/tools/TensorFold-moe
export TENSORFOLD_NO_UPDATE_CHECK=1 CUDA_HOME=/opt/cuda PATH=/opt/cuda/bin:$PATH
up() { for i in $(seq 1 240); do curl -sf http://127.0.0.1:$1/v1/models > /dev/null && return 0; sleep 2; done; return 1; }
O=~/.cache/r1s-zig/prof/pf-$1
if [ $1 = zig ]; then
  cd $TF
  setsid nsys profile -t cuda --cuda-graph-trace=graph --export sqlite -o $O -f true ./zig-out/native/bin/tensorfold-native serve ~/models/ornith/r1s --context 65536 --parallel 1 --port 8095 --backend cuda --no-update-check > $O.log 2>&1 < /dev/null &
  up 8095 || { tail $O.log; exit 1; }
  MODEL_NAME=r1s python3 $H/longctx.py 8095 16000; MODEL_NAME=r1s python3 $H/longctx.py 8095 16000
  for p in $(ps -C tensorfold-nati -o pid=); do kill $p; done
else
  cd $A/envs/tensorfold060
  TENSORFOLD_CUDA_RESERVE_GIB=1 TF_GPU_ONLY_BUDGET=1 setsid nsys profile -t cuda --cuda-graph-trace=graph --export sqlite -o $O -f true uv run python $H/serve_kernels.py new serve ~/models/ornith/r1s --name ornith --context 65536 --port 8090 --prompt-cache-gib 0 > $O.log 2>&1 < /dev/null &
  up 8090 || { tail $O.log; exit 1; }
  MODEL_NAME=ornith python3 $H/longctx.py 8090 16000; MODEL_NAME=ornith python3 $H/longctx.py 8090 16000
  for p in $(pgrep -f "serve_kernels.py new serve"); do kill -INT $p; done
fi
for i in $(seq 120); do [ -f $O.sqlite ] && ! pgrep -x nsys > /dev/null && break; sleep 2; done
ls -la $O.sqlite
