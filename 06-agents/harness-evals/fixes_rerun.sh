#!/bin/bash
# fixes_rerun.sh: rerun the fix test's lost rollouts (sandbox, server or harness errors, a hung run, the 16:45 wifi
# drop) so both sides end with about the same number of usable runs. Submits four servers at once (they take ~15
# minutes to start), waits for the last fix-test driver (fixes-on-p1b) to finish, then has lost_runs.py write the task
# files: two servers per side, one rollout per task each (tasks that lost both tries are on both servers). Same
# settings as fixes_full.sh; labels fixes-{off,on}-r{1,2}, ports 8116-8119, 5h cap, each server cancels itself when done.
cd /home/riels/Projects/Personal/Research/agent-training/06-agents/harness-evals
declare -A JOB
for arm in off on; do
  for s in 1 2; do
    J=$(TIMEOUT=5h ./hf_serve.sh submit sd-out/r1s-sd ornith35b-r1s-sd-fixes-$arm-r$s 2>&1 | tail -1)
    if ! [[ $J =~ ^[0-9a-f]{24}$ ]]; then echo "$(date +%T) submit failed for $arm r$s: $J"; continue; fi
    JOB[$arm$s]=$J
    echo "$(date -Iseconds) START hfjob $J a10g-large fixes $arm r$s (rerun of lost rollouts), 5h cap, auto-cancel" >> /tmp/prime-spend.log
    echo "$(date +%T) fixes $arm r$s: job $J"
  done
done
while ps -eo args | grep -q '^/bin/bash ./[f]ixes_run.sh fixes-on-p1b'; do sleep 30; done
echo "$(date +%T) fixes-on-p1b done"
python3 lost_runs.py
i=0
for arm in off on; do
  for s in 1 2; do
    i=$((i + 1)); J=${JOB[$arm$s]}; T=tb2_lost_${arm}_s$s.txt
    [ -n "$J" ] || continue
    if ! [ -s $T ]; then
      uvx -q --from huggingface_hub hf jobs cancel $J > /dev/null 2>&1
      echo "$(date -Iseconds) TERMINATE hfjob $J (fixes $arm r$s: nothing to rerun)" >> /tmp/prime-spend.log; continue
    fi
    F=cutoff,check,image; [ $arm = off ] && F=none
    setsid nohup ./fixes_run.sh fixes-$arm-r$s $J $F $T 1 $((8115 + i)) > logs/fixes-fixes-$arm-r$s.out 2>&1 < /dev/null &
    echo "$(date +%T) fixes $arm r$s: job $J, port $((8115 + i)), $(grep -c . $T) tasks"
  done
done
