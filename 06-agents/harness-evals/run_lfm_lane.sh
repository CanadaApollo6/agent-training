#!/bin/bash
# One lane of the LFM2.5 Terminal-Bench 2 run: pi on a share of tasks.txt, one conversation at a time, against one
# server (M3TiS's serves one stream; interleaved conversations would re-read each other's histories). Sandboxes on
# Prime VMs; the server is tunnelled to 127.0.0.1:<port>. Same settings as run_pilot.sh otherwise.
#
#   ./run_lfm_lane.sh <label, e.g. lfm25-m3tis-a1l1> <port> <first task index> <count> [concurrency]
set -euo pipefail
cd "$(dirname "$0")"
LABEL=$1; PORT=$2; FIRST=$3; COUNT=$4; CONC=${5:-1}
TASKS=$(python3 -c "import json,sys; print(json.dumps(open('tasks.txt').read().split()[$FIRST:$FIRST+$COUNT]))")
export LOCAL_KEY=none
mkdir -p logs
uv run vf-eval primeintellect/terminal-bench-2 -m "$LABEL" \
  --client.base-url http://127.0.0.1:$PORT/v1 --client.api-key-var LOCAL_KEY \
  --env.agent.harness.id pi --env.agent.runtime.type prime --env.taskset.tasks "$TASKS" \
  --env.agent.max-turns 60 --env.agent.timeout.rollout ${TIMEOUT:-3600} \
  -n $COUNT -r 1 -c "$CONC" --no-push --no-rich > "logs/$LABEL-pi.log" 2>&1
grep -E "rollout done" "logs/$LABEL-pi.log" | sed -E 's/.*task=([0-9]+) reward=([0-9.]+) turns=([0-9]+) stop=([a-z_]+).*/\1 \2 \3 \4/'
