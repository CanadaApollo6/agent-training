#!/bin/bash
# watch.sh JOB RUN: every 60s pull the run's logs from the rl-out bucket to /tmp/rl-out/RUN; print new job.log lines,
# orchestrator step lines and errors, and the job's terminal status. Exits when the job ends.
J=$1 RUN=$2 D=/tmp/rl-out/$RUN; HF="uvx -q --from huggingface_hub hf"; mkdir -p $D; seen=0; seen_err=0; seen_step=0
while true; do
  $HF buckets sync "hf://buckets/CanadaApollo6/rl-out/$RUN" $D > /dev/null 2>&1 || true
  if [ -f $D/job.log ]; then n=$(wc -l < $D/job.log); [ $n -gt $seen ] && tail -n +$((seen+1)) $D/job.log; seen=$n; fi
  steps=$(grep -rhE 'Step [0-9]+ \|' $D/outputs 2>/dev/null | grep -i orchestr -m 50 | wc -l)
  errs=$(cat $D/rl.stdout $(find $D/outputs -name '*.log' 2>/dev/null) 2>/dev/null | grep -cE 'Traceback|CUDA out of memory|OutOfMemory|FAILED|Error:')
  [ $errs -gt $seen_err ] && echo "new error lines: $errs (was $seen_err)" && seen_err=$errs
  st=$($HF jobs inspect $J 2>/dev/null | grep -oE "'stage': '[A-Z_]+'" | cut -d"'" -f4)
  [ "$st" != "$last_st" ] && echo "job stage: $st" && last_st=$st
  case $st in COMPLETED|ERROR|CANCELED|DELETED) echo "job $st"; exit 0;; esac
  sleep 60
done
