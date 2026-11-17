#!/bin/bash
# On the pod, after setup_pod.sh: build the pi adapter's 4-bit weights (CPU) while GPU 0 measures the Vontra build's
# speed, then the KL check on all four GPUs, then the new build's speed on GPU 0. Expects ~/pod (these scripts and
# kl.jsonl, turns.jsonl). Writes ~/out; progress lines in ~/out/run.log end with RUN_DONE or RUN_FAILED.
set -uo pipefail
O=~/out; P=~/pod; mkdir -p $O
log() { echo "$(date +%T) $*" | tee -a $O/run.log; }
[ -f ~/compat_path ] && export LD_LIBRARY_PATH=$(cat ~/compat_path)
export PATH=~/kl/bin:$PATH  # m3tis-serve.sh checks for the hf CLI
"$HOME/m3tis/bin/tensorfold-native" --version > $O/tf-version.txt 2>&1

speed() {  # label, weights dir
  local log=$O/serve-$1.log
  CUDA_VISIBLE_DEVICES=0 M3TIS_MODEL=$2 M3TIS_DRAFTER=~/drafter CONTEXT=32768 PORT=8080 \
    ~/m3tis/m3tis-serve.sh --no-thinking > $log 2>&1 & local pid=$!
  for _ in $(seq 600); do
    curl -sf 127.0.0.1:8080/v1/models > /dev/null && break
    kill -0 $pid 2> /dev/null || { log "speed $1: server died, see $log"; return 1; }
    sleep 1
  done
  SERVER_LOG=$log ~/kl/bin/python $P/speed_q27.py 8080 $O/speed-$1.json $P/turns.jsonl > $O/speed-$1.out 2>&1
  local rc=$?
  kill $pid; wait $pid 2> /dev/null
  log "speed $1 exit $rc"
  return $rc
}

log "start"
( cd $P && ~/kl/bin/python build_mlx4.py --base ~/qwen27 --lora ~/pi-lora --vontra ~/vontra --out ~/q27pi-mlx4 \
    > $O/build.log 2>&1; echo $? > $O/build.rc ) &
BUILD=$!
speed vontra ~/vontra
wait $BUILD
log "build exit $(cat $O/build.rc)"
[ "$(cat $O/build.rc)" = 0 ] || { log RUN_FAILED; exit 1; }
cp ~/q27pi-mlx4/M3TIS_BUILD.json $O/

( cd $P && ~/kl/bin/python kl_q27.py --base ~/qwen27 --lora ~/pi-lora --vontra ~/vontra --adapted ~/q27pi-mlx4 \
    --texts $P/kl.jsonl --out $O/kl.json > $O/kl.log 2>&1 )
log "kl exit $?"
speed pi ~/q27pi-mlx4
log RUN_DONE
