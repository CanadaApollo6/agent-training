#!/usr/bin/env bash
# pod_conc_base.sh: runs on the pod; the base binary's concurrency, drafted-vs-plain and resume runs, as new's
S=$HOME/scripts
B=$HOME/base/tensorfold-native
CONC=1 REPS=1 TEMPS=0 $S/pod_serve_bench.sh $B $HOME/models/r1s conc-base-r1s --drafter $HOME/models/r1s --context 8192 --parallel 2
CONC=1 REPS=1 TEMPS=0 $S/pod_serve_bench.sh $B $HOME/models/q27 conc-base-q27 --drafter $HOME/models/dflash2 --context 4096 --parallel 2
echo ALL_DONE
