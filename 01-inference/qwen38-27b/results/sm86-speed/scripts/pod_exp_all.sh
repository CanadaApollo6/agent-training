#!/usr/bin/env bash
# pod_exp_all.sh: runs on the pod; piece B (experts) engine checks and served R1s benches against base
S=$HOME/scripts
B=$HOME/base/tensorfold-native
N=$HOME/tf-exp/zig-out/native/bin/tensorfold-native
R1S=(--drafter $HOME/models/r1s --context 8192)
$S/pod_engine_runs.sh exp $HOME/tf-exp/zig-out/bin/tf-qwen35-cuda r1s
$S/pod_engine_runs.sh base2 $HOME/base/tf-qwen35-cuda r1s
$S/pod_serve_bench.sh $B $HOME/models/r1s base2-r1s "${R1S[@]}" --parallel 1
$S/pod_serve_bench.sh $N $HOME/models/r1s exp-r1s "${R1S[@]}" --parallel 1
$S/pod_serve_bench.sh $N $HOME/models/r1s exp-plain-r1s --drafter none --context 8192 --parallel 1
$S/pod_serve_bench.sh $B $HOME/models/r1s base2-plain-r1s --drafter none --context 8192 --parallel 1
CONC=1 REPS=1 TEMPS=0 $S/pod_serve_bench.sh $N $HOME/models/r1s conc-exp-r1s "${R1S[@]}" --parallel 2
echo ALL_DONE
