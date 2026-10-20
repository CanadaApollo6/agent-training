#!/usr/bin/env bash
# pod_serve_bench.sh BIN MODEL LABEL [serve args...]: runs on the pod. Serves BIN on port 8097, runs bench_openai,
# then (CONC=1) bench_concurrent --alone --serial and bench_cache (fresh vs resumed), and stops the server.
# Env: TEMPS (0,1.0), TOKENS (128), REPS (3), CONC (0), TOOLS (~/tf-new/tools), OUT (~/runs/serve)
set -uo pipefail
. ~/env.sh
BIN=$1; MODEL=$2; LABEL=$3; shift 3
TOOLS=${TOOLS:-$HOME/tf-new/tools}
OUT=${OUT:-$HOME/runs/serve}
PORT=8097
PY=${PY:-$(uv python find 3.12)}
mkdir -p "$OUT"
# wait for the previous server to release the GPU
for i in $(seq 1 120); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 500 ] && break; sleep 1; done
LOG="$OUT/serve-$LABEL.log"
"$BIN" serve "$MODEL" --port $PORT --backend cuda --no-update-check "$@" > "$LOG" 2>&1 &
SPID=$!
for i in $(seq 1 600); do
  curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1 && break
  if ! kill -0 $SPID 2>/dev/null; then echo "server died"; tail -20 "$LOG"; exit 1; fi
  sleep 1
done
NAME=$(curl -s http://127.0.0.1:$PORT/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')
curl -s http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' \
  -d "{\"model\":\"$NAME\",\"prompt\":\"Hello\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
"$PY" "$TOOLS/bench_openai.py" http://127.0.0.1:$PORT "$NAME" --tokens "${TOKENS:-128}" --reps "${REPS:-3}" \
  --temperatures "${TEMPS:-0,1.0}" --label "$LABEL" --output "$OUT/openai-$LABEL.json" 2>&1 | tail -12
if [ "${CONC:-0}" = 1 ]; then
  "$PY" "$TOOLS/bench_concurrent.py" http://127.0.0.1:$PORT "$NAME" --tokens 128 --reps 2 --temperatures 0,1.0 \
    --levels "${LEVELS:-1,2}" --alone --serial --output "$OUT/concurrent-$LABEL.json" 2>&1 | tail -12
  (cd "$TOOLS" && "$PY" bench_cache.py http://127.0.0.1:$PORT "$NAME" --tokens 64 --repeats 2 --streams 2 \
    --output "$OUT/cache-$LABEL.json" 2>&1 | tail -12)
fi
kill -INT $SPID
wait $SPID 2>/dev/null
sleep 2
echo "done $LABEL"
