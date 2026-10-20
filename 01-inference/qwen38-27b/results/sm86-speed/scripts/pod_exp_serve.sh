#!/usr/bin/env bash
# pod_exp_serve.sh: runs on the pod; piece B served R1s, base vs exp, plain and MTP drafts, then exp concurrency
S=$HOME/scripts
B=$HOME/base/tensorfold-native
N=$HOME/tf-exp/zig-out/native/bin/tensorfold-native
D=(--drafter $HOME/models/r1s --context 8192 --parallel 1)
$S/pod_serve_bench.sh $N $HOME/models/r1s exp-plain-r1s --context 8192 --parallel 1
$S/pod_serve_bench.sh $B $HOME/models/r1s base-plain-r1s --context 8192 --parallel 1
$S/pod_serve_bench.sh $N $HOME/models/r1s exp-r1s "${D[@]}"
$S/pod_serve_bench.sh $B $HOME/models/r1s base3-r1s "${D[@]}"
CONC=1 REPS=1 TEMPS=0 $S/pod_serve_bench.sh $N $HOME/models/r1s conc-exp-r1s --drafter $HOME/models/r1s --context 8192 --parallel 2
echo ALL_DONE
