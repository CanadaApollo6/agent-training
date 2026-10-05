#!/bin/bash
# lessons_resume.sh H JOB: resume the lessons loop half H against the server on HF job JOB, cancel the job after.
cd /home/riels/Projects/Personal/Research/agent-training/06-agents/harness-evals
H=$1 J=$2
export HF_TOKEN=$(cat ~/.cache/huggingface/token); U=https://$J--8080.hf.jobs/v1
echo "$(date +%T) resuming at attempt 4 on $J (clamp on)"
uv run python lessons_loop.py --label ornith35b-r1s-sd --name sd-lessons-h$H --tasks datagen/lessons_half$H.txt --attempts 6 --conc 4 --base-url $U --key-var HF_TOKEN
echo "$(date +%T) loop exit $?"; uvx -q --from huggingface_hub hf jobs cancel $J
echo "$(date -Iseconds) TERMINATE hfjob $J (lessons h$H done)" >> /tmp/prime-spend.log
