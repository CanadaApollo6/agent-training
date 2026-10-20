#!/usr/bin/env bash
# serve.sh LABEL OUTDIR STEPS [serve args...]: under the GPU lock, serve on 8096, run STEPS (comma list), stop.
# Steps: openai, conc, cache, long:<depths separated by +>, smi. BIN overrides the binary.
set -u
set -o pipefail
[ -d /usr/local/cuda/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
LABEL=$1; OUT=$2; STEPS=$3; shift 3
TF=$HOME/tf-kv4
PY="$HOME/.local/bin/uv run -q --no-project --python 3.12 python"
BIN=${BIN:-$TF/zig-out/native/bin/tensorfold-native}
MODEL=$HOME/.cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c
DRAFT=$HOME/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4
PORT=8096
mkdir -p "$OUT"


echo "lock taken $(date +%T)"
curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1 && { echo "port $PORT busy"; exit 1; }
LOG="$OUT/serve-$LABEL.log"
"$BIN" serve "$MODEL" --port $PORT --backend cuda --no-update-check --drafter "$DRAFT" "$@" > "$LOG" 2>&1 &
SPID=$!
( while kill -0 $SPID 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > "$OUT/smi-$LABEL.txt" &
MON=$!
for i in $(seq 1 300); do
  curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1 && break
  if ! kill -0 $SPID 2>/dev/null; then echo "server died"; tail -20 "$LOG"; kill $MON; exit 1; fi
  sleep 1
done
grep -E "startup|serving|cache|GiB" "$LOG" | head -8
NAME=$(curl -s http://127.0.0.1:$PORT/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')
curl -s http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' \
  -d "{\"model\":\"$NAME\",\"prompt\":\"Hello\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
echo "idle device MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
IFS=, read -ra S <<< "$STEPS"
for s in "${S[@]}"; do
  t0=$(date +%s)
  case $s in
    openai) $PY $TF/tools/bench_openai.py http://127.0.0.1:$PORT "$NAME" --tokens 128 --reps 3 --temperatures 0,1.0 \
              --label "$LABEL" --output "$OUT/openai-$LABEL.jsonl" 2>&1 | tail -6 ;;
    conc) $PY $TF/tools/bench_concurrent.py http://127.0.0.1:$PORT "$NAME" --tokens 128 --reps 2 --temperatures 0,1.0 \
              --levels "${LEVELS:-1,2}" --alone --serial --output "$OUT/concurrent-$LABEL.jsonl" 2>&1 | tail -8 ;;
    cache) $PY $TF/tools/bench_cache.py http://127.0.0.1:$PORT "$NAME" --require-cold --output "$OUT/cache-$LABEL.json" 2>&1 | tail -6 ;;
    long:*) d=${s#long:}; $PY $HOME/longctx.py http://127.0.0.1:$PORT "$NAME" \
              $HOME/fixture-full.json "$OUT/long-$LABEL.json" ${d//+/ } 2>&1 | tail -8 ;;
    probe:*) p=${s#probe:}; $PY $HOME/probe.py http://127.0.0.1:$PORT "$NAME" $HOME/fixture-64k.json ${p//+/ } \
              2>&1 | tee -a "$OUT/probe-$LABEL.jsonl" | cut -c1-600 ;;
  esac
  echo "step $s took $(( $(date +%s) - t0 )) s"
done
for pid in $(ps -C tensorfold-nati -o pid=,args= | awk -v p="--port $PORT" 'index($0,p){print $1}'); do kill -INT $pid; done
wait $SPID 2>/dev/null
kill $MON 2>/dev/null
echo "peak device MiB: $(sort -n "$OUT/smi-$LABEL.txt" | tail -1)"
echo "done $LABEL $(date +%T)"
