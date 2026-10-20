#!/usr/bin/env bash
# serve_bench.sh BIN MODEL LABEL OUTDIR [serve args...]
# Starts BIN serve on port 8097 under the shared GPU lock, runs bench_openai (and optional extras), stops the server.
# Env: TEMPS (default 0,1.0), TOKENS (128), REPS (3), NSYS=1 (trace the bench window), TIMING=1 (TF_TIMING), CONC=1
set -uo pipefail
BIN=$1; MODEL=$2; LABEL=$3; OUT=$4; shift 4
TF=/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-speed
PORT=8097
mkdir -p "$OUT"
exec 9>/tmp/rtx3090.lock
flock 9
LOG="$OUT/serve-$LABEL.log"
envs=()
[ "${TIMING:-0}" = 1 ] && envs+=(TF_TIMING=1)
if [ "${NSYS:-0}" = 1 ]; then
  env "${envs[@]}" nsys launch --trace=cuda,nvtx --cuda-graph-trace=node --session-new="sm86$LABEL" \
    "$BIN" serve "$MODEL" --port $PORT --backend cuda --no-update-check "$@" > "$LOG" 2>&1 &
else
  env "${envs[@]}" "$BIN" serve "$MODEL" --port $PORT --backend cuda --no-update-check "$@" > "$LOG" 2>&1 &
fi
SPID=$!
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1 && break
  if ! kill -0 $SPID 2>/dev/null; then echo "server died"; tail -20 "$LOG"; exit 1; fi
  sleep 1
done
NAME=$(curl -s http://127.0.0.1:$PORT/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')
# one short warm request so the first measured round pays no first-launch costs
curl -s http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' \
  -d "{\"model\":\"$NAME\",\"prompt\":\"Hello\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
[ "${NSYS:-0}" = 1 ] && nsys start --session="sm86$LABEL" -o "$OUT/nsys-$LABEL" --force-overwrite=true
python3 $TF/tools/bench_openai.py http://127.0.0.1:$PORT "$NAME" --tokens "${TOKENS:-128}" --reps "${REPS:-3}" \
  --temperatures "${TEMPS:-0,1.0}" --label "$LABEL" --output "$OUT/openai-$LABEL.json" 2>&1 | tail -20
[ "${NSYS:-0}" = 1 ] && nsys stop --session="sm86$LABEL"
if [ "${CONC:-0}" = 1 ]; then
  python3 $TF/tools/bench_concurrent.py http://127.0.0.1:$PORT "$NAME" --tokens 128 --reps 2 --temperatures 0,1.0 \
    --levels "${LEVELS:-1,2}" --alone --serial --output "$OUT/concurrent-$LABEL.json" 2>&1 | tail -15
fi
# stop only our server: the process listening on our port
for pid in $(ps -C tensorfold-nati -o pid=,args= | awk -v p="--port $PORT" 'index($0,p){print $1}'); do kill -INT $pid; done
[ "${NSYS:-0}" = 1 ] && nsys shutdown --session="sm86$LABEL" > /dev/null 2>&1
wait $SPID 2>/dev/null
sleep 2
echo "done $LABEL"
