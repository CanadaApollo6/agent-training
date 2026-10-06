#!/bin/bash
# stop_when_done.sh LOG LAST_INDEX ROLLOUTS DRIVER_PID JOB: when the run logging to LOG has finished every rollout of
# tasks 0..LAST_INDEX (task= in its "rollout done" lines follows the tasks-file order), stop its fixes_run.sh driver
# (and everything under it: run_eval, vf-eval, fix_proxy), cancel its HF job, and delete the Prime sandboxes its
# in-flight rollouts left behind. Used when a second server takes the tail of a task list that the first server
# would not finish before its job limit; the first server's rollouts past LAST_INDEX are duplicates.
cd /home/riels/Projects/Personal/Research/agent-training/06-agents/harness-evals
LOG=$1 LAST=$2 R=$3 D=$4 J=$5
need=$(( (LAST + 1) * R ))
until [ "$(grep -oE 'rollout done: id=\w+ task=[0-9]+' $LOG | awk -F'task=' -v m=$LAST '$2 <= m' | wc -l)" -ge $need ]; do
    kill -0 $D 2>/dev/null || { echo "$(date +%T) driver $D gone before tasks 0..$LAST finished"; exit 1; }
    sleep 60
done
desc() { echo $1; for c in $(pgrep -P $1); do desc $c; done; }
kill -9 $(desc $D)
echo "$(date +%T) $LOG: tasks 0..$LAST done; stopped driver $D"
uvx -q --from huggingface_hub hf jobs cancel $J > /dev/null 2>&1
echo "$(date -Iseconds) TERMINATE hfjob $J (tasks 0..$LAST done; the tail runs on another server)" >> /tmp/prime-spend.log
sleep 5
uvx -q --from prime prime sandbox list --output json 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin); d = d.get('sandboxes', d) if isinstance(d, dict) else d
text = open('$LOG').read()
print(' '.join(s['id'] for s in d if s['name'] in text or s['id'] in text))" | xargs -r -n1 uvx -q --from prime prime sandbox delete --yes
echo "$(date +%T) deleted its leftover sandboxes"
