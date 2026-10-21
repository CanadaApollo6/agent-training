#!/usr/bin/env bash
# pod_alt27.sh TAG ROUNDS NAME=BIN...: runs on the pod; the 27B with DFlash2 drafts: bench_openai T=0/T=1 at
# --parallel 1, then the CONC sequence at --parallel 2, each binary in turn, ROUNDS times; per-request hashes in order.
TAG=$1; ROUNDS=$2; shift 2
OUT=$HOME/runs/alt-$TAG; mkdir -p $OUT
D=(--drafter $HOME/models/dflash2 --context 4096)
sum() { grep '"streams"' $1 | python3 -c '
import sys,json
a=[0,0];s=[0,0]
for l in sys.stdin:
  r=json.loads(l)
  for k,acc in (("alone",a),("serial",s)):
    if r.get(k): acc[0]+=r[k]["equal"]; acc[1]+=r[k]["unequal"]
print("   alone eq/uneq",a,"serial eq/uneq",s)'; }
hs() { grep 'done cmpl' $1 | sed -E 's/.*prompt=([0-9]+).*sha=(\w{8}).*accepted=([0-9/]+).*/\1:\2:\3/' | tr '\n' ' '; }
for r in $(seq 1 $ROUNDS); do
  for nb in "$@"; do
    n=${nb%%=*}; b=${nb#*=}
    OUT=$OUT $HOME/scripts/pod_serve_bench.sh $b $HOME/models/q27 p1-$n-$r "${D[@]}" --parallel 1 > $OUT/bench-p1-$n-$r.log 2>&1
    echo "p1-$n-$r $(hs $OUT/serve-p1-$n-$r.log)"
    CONC=1 REPS=1 TEMPS=0 OUT=$OUT $HOME/scripts/pod_serve_bench.sh $b $HOME/models/q27 p2-$n-$r "${D[@]}" --parallel 2 > $OUT/bench-p2-$n-$r.log 2>&1
    echo "p2-$n-$r $(hs $OUT/serve-p2-$n-$r.log)"; sum $OUT/bench-p2-$n-$r.log
  done
done
echo ALT_DONE
