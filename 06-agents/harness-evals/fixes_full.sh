#!/bin/bash
# fixes_full.sh [START HH:MM]: the bigger test of the harness fixes. Waits until START (default 07:00; "now" skips),
# then submits four R1s-SD servers and runs fixes_run.sh on each: TB2 without the two qemu tasks (87), halves p1/p2,
# 2 tries per task, fixes off (pass through) and fixes on (cutoff,check,image) on the same halves. Logs to
# logs/fixes-full.out; each server cancels itself when its half is done.
cd /home/riels/Projects/Personal/Research/agent-training/06-agents/harness-evals
START=${1:-07:00}
if [ "$START" != now ]; then
  t=$(date -d "$START" +%s); [ "$t" -le "$(date +%s)" ] && t=$(date -d "tomorrow $START" +%s)
  echo "$(date +%T) waiting until $(date -d @$t)"; sleep $(( t - $(date +%s) ))
fi
i=0
for arm in off on; do
  for p in 1 2; do
    i=$((i + 1))
    J=$(TIMEOUT=12h ./hf_serve.sh submit sd-out/r1s-sd ornith35b-r1s-sd-fixes-$arm-p$p 2>&1 | tail -1)
    if ! [[ $J =~ ^[0-9a-f]{24}$ ]]; then echo "$(date +%T) submit failed for $arm p$p: $J"; continue; fi
    echo "$(date -Iseconds) START hfjob $J a10g-large fixes $arm p$p (TB2 43-44 tasks x 2), 12h cap, auto-cancel" >> /tmp/prime-spend.log
    F=cutoff,check,image; [ $arm = off ] && F=none
    setsid nohup ./fixes_run.sh fixes-$arm-p$p $J $F tb2_noqemu_p$p.txt 2 $((8110 + i)) > logs/fixes-fixes-$arm-p$p.out 2>&1 < /dev/null &
    echo "$(date +%T) fixes $arm p$p: job $J, port $((8110 + i))"
  done
done
