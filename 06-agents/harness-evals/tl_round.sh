#!/bin/bash
# tl_round.sh PART JOB: Terminal-Lego set "hard", part PART (results/taskset_hard.partPART.txt), 2 tries per task at a
# 32K reply cap against the R1s-SD server on HF job JOB (waits for it), then cancels the job.
cd /home/riels/Projects/Personal/Research/agent-training/06-agents/harness-evals
P=$1 J=$2 L=ornith35b-r1s-sd-tlhard$1
export HF_TOKEN=$(cat ~/.cache/huggingface/token); U=https://$J--8080.hf.jobs/v1
log() { echo "$(date +%T) $*" >> logs/hf-series.log; }
for i in $(seq 180); do curl -sf -m 10 -H "Authorization: Bearer $HF_TOKEN" $U/models > /dev/null && break; sleep 20; done
log "$L prime_agent start (job $J)"
TL_SET=hard TASKS_FILE=../../04-post-training/terminal-lego/results/taskset_hard.part$P.txt ROLLOUTS=2 \
  EXTRA="--sampling.max-tokens 32768" BASE_URL=$U KEY_VAR=HF_TOKEN ./run_eval.sh tl prime_agent $L 4 > /dev/null 2>&1
log "$L prime_agent done (exit $?): $(grep -c 'reward=1' logs/tl-$L-prime_agent.log) solved of $(grep -c 'rollout done' logs/tl-$L-prime_agent.log)"
uvx -q --from huggingface_hub hf jobs cancel $J > /dev/null 2>&1; log "$L: cancelled job $J"
echo "$(date -Iseconds) TERMINATE hfjob $J (tl hard part $P done)" >> /tmp/prime-spend.log
