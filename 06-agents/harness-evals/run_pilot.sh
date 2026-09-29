#!/bin/bash
# One harness x one model on the 20-task Terminal-Bench 2 pilot (tasks.txt), sandboxes on Prime VMs.
# The model is served by llama-server on the Prime pod, forwarded to 127.0.0.1:8000 over SSH.
#
#   ./run_pilot.sh <harness: pi|prime_agent> <label, e.g. ornith9b-q8> [concurrency]
set -euo pipefail
cd "$(dirname "$0")"
HARNESS=$1; LABEL=$2; CONC=${3:-4}
TASKS=$(python3 -c "import json; print(json.dumps(open('tasks.txt').read().split()))")
export LOCAL_KEY=none
mkdir -p logs
uv run vf-eval primeintellect/terminal-bench-2 -m "$LABEL" \
  --client.base-url http://127.0.0.1:8000/v1 --client.api-key-var LOCAL_KEY \
  --env.agent.harness.id "$HARNESS" --env.agent.runtime.type prime --env.taskset.tasks "$TASKS" \
  --env.agent.max-turns 60 --env.agent.timeout.rollout 1800 \
  -n 20 -r 1 -c "$CONC" --no-push --no-rich > "logs/$LABEL-$HARNESS.log" 2>&1
grep -E "rollout done" "logs/$LABEL-$HARNESS.log" | sed -E 's/.*task=([0-9]+) reward=([0-9.]+) turns=([0-9]+) stop=([a-z_]+).*/\1 \2 \3 \4/'
