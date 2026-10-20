#!/usr/bin/env bash
# pod_serve_all.sh: runs on the pod; base vs new served benches for R1s and the 27B, then new's exactness runs
S=$HOME/scripts
B=$HOME/base/tensorfold-native
N=$HOME/tf-new/zig-out/native/bin/tensorfold-native
R1S=(--drafter $HOME/models/r1s --context 8192)
Q27=(--drafter $HOME/models/dflash2 --context 4096)
$S/pod_serve_bench.sh $B $HOME/models/r1s base-r1s "${R1S[@]}" --parallel 1
$S/pod_serve_bench.sh $N $HOME/models/r1s new-r1s "${R1S[@]}" --parallel 1
$S/pod_serve_bench.sh $B $HOME/models/q27 base-q27 "${Q27[@]}" --parallel 1
$S/pod_serve_bench.sh $N $HOME/models/q27 new-q27 "${Q27[@]}" --parallel 1
CONC=1 REPS=1 TEMPS=0 $S/pod_serve_bench.sh $N $HOME/models/r1s conc-new-r1s "${R1S[@]}" --parallel 2
CONC=1 REPS=1 TEMPS=0 $S/pod_serve_bench.sh $N $HOME/models/q27 conc-new-q27 "${Q27[@]}" --parallel 2
echo ALL_DONE
