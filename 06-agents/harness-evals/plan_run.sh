#!/bin/bash
# plan_run.sh ARM JOB: plan_eval.py's arm ARM against the R1s-SD server on HF job JOB (waits for it), then cancels the job.
cd /home/riels/Projects/Personal/Research/agent-training/06-agents/harness-evals
ARM=$1 J=$2
export HF_TOKEN=$(cat ~/.cache/huggingface/token); U=https://$J--8080.hf.jobs/v1
for i in $(seq 180); do curl -sf -m 10 -H "Authorization: Bearer $HF_TOKEN" $U/models > /dev/null && break; sleep 20; done
echo "$(date +%T) server $J up"
uv run python plan_eval.py --arm $ARM --label ornith35b-r1s-sd --name sd-plan-$ARM --base-url $U --key-var HF_TOKEN
echo "$(date +%T) plan_eval exit $?"; uvx -q --from huggingface_hub hf jobs cancel $J
echo "$(date -Iseconds) TERMINATE hfjob $J (plan $ARM done)" >> /tmp/prime-spend.log
