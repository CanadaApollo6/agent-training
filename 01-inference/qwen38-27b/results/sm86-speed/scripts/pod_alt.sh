#!/usr/bin/env bash
# pod_alt.sh TAG ROUNDS NAME=BIN...: runs on the pod; the CONC served sequence (R1s, MTP drafts, --parallel 2) for
# each binary in turn, ROUNDS times, then each run's per-request hashes in order (prompt:sha:accepted).
TAG=$1; ROUNDS=$2; shift 2
OUT=$HOME/runs/alt-$TAG; mkdir -p $OUT
for r in $(seq 1 $ROUNDS); do
  for nb in "$@"; do
    n=${nb%%=*}; b=${nb#*=}
    CONC=1 REPS=1 TEMPS=0 OUT=$OUT $HOME/scripts/pod_serve_bench.sh $b $HOME/models/r1s $n-$r --drafter $HOME/models/r1s --context 8192 --parallel 2 > $OUT/bench-$n-$r.log 2>&1
    echo "$n-$r $(grep 'done cmpl' $OUT/serve-$n-$r.log | sed -E 's/.*prompt=([0-9]+).*sha=(\w{8}).*accepted=([0-9/]+).*/\1:\2:\3/' | tr '\n' ' ')"
    grep '"streams"' $OUT/bench-$n-$r.log | python3 -c '
import sys,json
a=[0,0];s=[0,0]
for l in sys.stdin:
  r=json.loads(l)
  for k,acc in (("alone",a),("serial",s)):
    if r.get(k): acc[0]+=r[k]["equal"]; acc[1]+=r[k]["unequal"]
print("   alone eq/uneq",a,"serial eq/uneq",s)'
  done
done
echo ALT_DONE
