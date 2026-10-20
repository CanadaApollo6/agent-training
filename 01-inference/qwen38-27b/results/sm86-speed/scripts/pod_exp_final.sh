#!/usr/bin/env bash
# pod_exp_final.sh: runs on the pod; piece B final build: engine checks and plain served R1s, base vs exp
S=$HOME/scripts
B=$HOME/base/tensorfold-native
N=$HOME/tf-exp/zig-out/native/bin/tensorfold-native
$S/pod_engine_runs.sh exp $HOME/tf-exp/zig-out/bin/tf-qwen35-cuda r1s
$S/pod_serve_bench.sh $B $HOME/models/r1s base-plain-r1s --context 8192 --parallel 1
$S/pod_serve_bench.sh $N $HOME/models/r1s exp-plain-r1s --context 8192 --parallel 1
$S/pod_serve_bench.sh $N $HOME/models/r1s exp-r1s --drafter $HOME/models/r1s --context 8192 --parallel 1
echo ALL_DONE
