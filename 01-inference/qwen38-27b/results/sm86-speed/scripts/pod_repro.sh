#!/usr/bin/env bash
# pod_repro.sh BIN LABEL: runs on the pod; R1s with MTP drafts at --parallel 2, bench_concurrent T=0 (levels 1,2,
# alone, serial), then the server log's per-request hashes in order and any SEGDBG lines. Env passes to the server.
. ~/env.sh
BIN=$1; LABEL=$2
OUT=$HOME/runs/repro; mkdir -p $OUT
PY=$(uv python find 3.12); PORT=8097; TOOLS=$HOME/tf-new/tools
for i in $(seq 1 120); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 500 ] && break; sleep 1; done
"$BIN" serve $HOME/models/r1s --port $PORT --backend cuda --no-update-check --drafter $HOME/models/r1s --context 8192 --parallel 2 > $OUT/serve-$LABEL.log 2>&1 &
SPID=$!
for i in $(seq 1 600); do curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1 && break; kill -0 $SPID 2>/dev/null || { echo died; tail $OUT/serve-$LABEL.log; exit 1; }; sleep 1; done
NAME=$(curl -s http://127.0.0.1:$PORT/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')
"$PY" $TOOLS/bench_concurrent.py http://127.0.0.1:$PORT "$NAME" --tokens 128 --reps 2 --temperatures 0 --levels 1,2 --alone --serial --output $OUT/conc-$LABEL.json > $OUT/bench-$LABEL.log 2>&1
kill -INT $SPID; wait $SPID 2>/dev/null; sleep 2
grep '"streams"' $OUT/bench-$LABEL.log | python3 -c '
import sys,json
a=[0,0];s=[0,0]
for l in sys.stdin:
  r=json.loads(l)
  for k,acc in (("alone",a),("serial",s)):
    if r.get(k): acc[0]+=r[k]["equal"]; acc[1]+=r[k]["unequal"]
print("alone eq/uneq",a,"serial eq/uneq",s)'
echo "$LABEL hashes: $(grep 'done cmpl' $OUT/serve-$LABEL.log | sed -E 's/.*prompt=([0-9]+).*sha=(\w{8}).*accepted=([0-9/]+).*/\1:\2:\3/' | tr '\n' ' ')"
grep -c SEGDBG $OUT/serve-$LABEL.log | sed "s/^/$LABEL SEGDBG lines: /"; grep -m3 SEGDBG $OUT/serve-$LABEL.log
