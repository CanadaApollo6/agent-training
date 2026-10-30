#!/usr/bin/env bash
# Release check on one 24 GB sm_86 card: the bundle at ~/m3tis serves, how big a window fits, the card's speed numbers,
# and plain greedy vs the exllamav3 reference. Expects ~/m3tis (bundle) and ~/pod (these scripts, oracle.json,
# p1-ids.json, long/*.txt prompts). Writes ~/out.
set -uo pipefail
B=~/m3tis; O=~/out; mkdir -p $O
nvidia-smi > $O/nvidia-smi.txt
"$B/bin/tensorfold-native" --version > $O/version.txt 2>&1

wait_up() {  # pid port log
  for _ in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$2/v1/models" >/dev/null && return 0
    kill -0 "$1" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}
peak() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }

for ctx in ${CONTEXTS:-131072 98304 65536}; do
  log=$O/fit-kv4-$ctx.log
  CONTEXT=$ctx PORT=8090 "$B/m3tis-serve.sh" > $log 2>&1 & pid=$!
  if wait_up $pid 8090; then
    echo "$ctx up $(peak) MiB" | tee -a $O/fit.txt
    kill $pid; wait $pid 2>/dev/null; break
  fi
  echo "$ctx failed" | tee -a $O/fit.txt; kill $pid 2>/dev/null; wait $pid 2>/dev/null
done
ctx=$(awk '/ up /{print $1; exit}' $O/fit.txt)

CONTEXT=$ctx PORT=8080 "$B/m3tis-serve.sh" > $O/serve-kv4.log 2>&1 & pid=$!
wait_up $pid 8080 || { echo "release server did not start"; exit 1; }
( while kill -0 $pid 2>/dev/null; do peak >> $O/mem-trace.txt; sleep 2; done ) &
SERVER_LOG=$O/serve-kv4.log python3 ~/pod/bench.py 8080 $O/bench-kv4.json ~/pod/long/*.txt | tee $O/bench-kv4.out
kill $pid; wait $pid 2>/dev/null
echo "peak $(sort -n $O/mem-trace.txt | tail -1) MiB" | tee -a $O/fit.txt

# greedy vs the exllamav3 reference's 128 tokens: plain (no draft, bf16 cache) must match all 128, as on the A100
greedy() {  # label, env..., -- extra serve args
  local label=$1; shift
  env "$@" CONTEXT=8192 PORT=8091 "$B/m3tis-serve.sh" --temperature 0 ${EXTRA:-} > $O/greedy-$label.log 2>&1 & local p=$!
  wait_up $p 8091 && echo "greedy $label $(python3 ~/pod/greedy_check.py 8091 ~/pod/p1-ids.json ~/pod/oracle.json $O/greedy-$label.json)" | tee -a $O/fit.txt
  kill $p; wait $p 2>/dev/null
}
EXTRA=--no-drafts greedy plain-kv16 TF_LLAMA_KV_BITS=16
greedy drafted-kv16 TF_LLAMA_KV_BITS=16
greedy drafted-kv4 TF_LLAMA_KV_BITS=4
