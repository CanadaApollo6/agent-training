#!/bin/bash
# fixes_run.sh NAME JOB FIXES TASKS_FILE [ROLLOUTS] [PORT]: TB2 tasks under prime_agent through fix_proxy.py (FIXES: a
# comma list of cutoff, check, image; "none" = pass through) against the R1s-SD server on HF job JOB (waits for it),
# 32K reply cap, 60 turns, 1-hour rollouts, 4 at a time. Labelled ornith35b-r1s-sd-NAME; the proxy logs every fix that
# fires to logs/fixes-NAME.jsonl. Then stops the proxy and cancels the job.
cd /home/riels/Projects/Personal/Research/agent-training/06-agents/harness-evals
N=$1 J=$2 F=$3 T=$4 R=${5:-1} P=${6:-8101} L=ornith35b-r1s-sd-$1
[ "$F" = none ] && F=""
export HF_TOKEN=$(cat ~/.cache/huggingface/token); U=https://$J--8080.hf.jobs
log() { echo "$(date +%T) $*" >> logs/hf-series.log; }
for i in $(seq 180); do curl -sf -m 10 -H "Authorization: Bearer $HF_TOKEN" $U/v1/models > /dev/null && break; sleep 20; done
python3 fix_proxy.py --upstream $U --key-var HF_TOKEN --port $P --fixes "$F" --log logs/fixes-$N.jsonl &
PROXY=$!
sleep 2
log "$L prime_agent start (job $J, fixes '${F:-none}', $(grep -c . $T) tasks x $R)"
TASKS_FILE=$T ROLLOUTS=$R EXTRA="--sampling.max-tokens 32768" BASE_URL=http://127.0.0.1:$P/v1 KEY_VAR=LOCAL_KEY \
  ./run_eval.sh tb2full prime_agent $L 4 > /dev/null 2>&1
log "$L prime_agent done (exit $?): $(grep -c 'reward=1' logs/tb2full-$L-prime_agent.log) solved of $(grep -c 'rollout done' logs/tb2full-$L-prime_agent.log)"
kill $PROXY
uvx -q --from huggingface_hub hf jobs cancel $J > /dev/null 2>&1; log "$L: cancelled job $J"
echo "$(date -Iseconds) TERMINATE hfjob $J (fixes $N done)" >> /tmp/prime-spend.log
