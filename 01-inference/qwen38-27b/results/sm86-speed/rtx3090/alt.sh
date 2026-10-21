#!/usr/bin/env bash
# alt.sh MODEL_TAG ROUNDS NAMES...: local 3090 alternation of served base vs branches (4 GiB GPU reserve kept free)
set -uo pipefail
TAG=$1; ROUNDS=$2; shift 2
R=/tmp/tf3090; OUT=$R/out/$TAG; mkdir -p $OUT; TOOLS=$R/base/tools; PORT=8097
PY=$(uv python find 3.12)
case $TAG in
  r1s) MODEL=$HOME/.cache/m3tis2/bundle; ARGS="--drafter $MODEL --context 8192 --parallel 2";;
  r1p) MODEL=$HOME/.cache/m3tis2/bundle; ARGS="--context 8192 --parallel 2";;
  q27) MODEL=$HOME/.cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c
       D=$(ls -d $HOME/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/*/ | head -1)
       ARGS="--drafter $D --context 4096 --parallel 2";;
esac
export TENSORFOLD_MEMORY_RESERVE_GIB=4
for r in $(seq 1 $ROUNDS); do for n in "$@"; do
  L=$n-$r; LOG=$OUT/serve-$L.log
  $R/$n/zig-out/native/bin/tensorfold-native serve $MODEL --port $PORT --backend cuda --no-update-check --prompt-cache-gib 2 $ARGS > $LOG 2>&1 &
  SPID=$!
  for i in $(seq 1 600); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; kill -0 $SPID 2>/dev/null || { echo "$L server died"; tail -5 $LOG; exit 1; }; sleep 1; done
  echo "$L up: $(nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader)"
  NAME=$(curl -s http://127.0.0.1:$PORT/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')
  curl -s http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' -d "{\"model\":\"$NAME\",\"prompt\":\"Hello\",\"max_tokens\":16,\"temperature\":0}" >/dev/null
  "$PY" $TOOLS/bench_openai.py http://127.0.0.1:$PORT "$NAME" --tokens 128 --reps 3 --temperatures 0,1.0 --label $L --output $OUT/openai-$L.json > $OUT/bench-$L.log 2>&1
  "$PY" $TOOLS/bench_concurrent.py http://127.0.0.1:$PORT "$NAME" --tokens 128 --reps 2 --temperatures 0,1.0 --levels 1,2 --alone --serial --output $OUT/concurrent-$L.json >> $OUT/bench-$L.log 2>&1
  (cd $TOOLS && "$PY" bench_cache.py http://127.0.0.1:$PORT "$NAME" --tokens 64 --repeats 2 --streams 2 --output $OUT/cache-$L.json >> $OUT/bench-$L.log 2>&1); echo "cache exit $?" >> $OUT/bench-$L.log
  kill -INT $SPID; wait $SPID 2>/dev/null; sleep 3
  echo "$L hashes: $(grep 'done cmpl' $LOG | sed -E 's/.*prompt=([0-9]+).*sha=(\w{8}).*/\1:\2/' | sort | uniq -c | awk '{print $2"x"$1}' | tr '\n' ' ')"
  grep '"streams"' $OUT/bench-$L.log | python3 -c '
import sys,json
a=[0,0];s=[0,0]
for l in sys.stdin:
  r=json.loads(l)
  for k,acc in (("alone",a),("serial",s)):
    if r.get(k): acc[0]+=r[k]["equal"]; acc[1]+=r[k]["unequal"]
print("   alone eq/uneq",a,"serial eq/uneq",s)'
  python3 -c 'import json,sys;print("   tok/s"," ".join("%s@T%s=%.1f" % (d["prompt"][:4],d["temperature"],d["decode_tps_median"]) for d in json.load(open(sys.argv[1]))))' $OUT/openai-$L.json
done; done
echo ALT_DONE
