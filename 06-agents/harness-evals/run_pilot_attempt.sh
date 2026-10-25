#!/bin/bash
# One attempt of the 19-task Terminal-Bench 2 pilot (tasks.txt minus qemu-startup) under pi, against an OpenAI-style
# server tunnelled to 127.0.0.1:<port>. Sandboxes on Prime VMs. Same caps as the R1s, LFM2.5 and Flint pilots: 60
# turns, 60 minutes. pi sends no sampling, so the server's defaults apply. Generalises run_flint_attempt.sh.
#
#   ./run_pilot_attempt.sh <label prefix> <attempt number> <port> [concurrency]   # label: <prefix>-a<attempt>
set -euo pipefail
cd "$(dirname "$0")"
P=$1; A=$2; PORT=$3; CONC=${4:-7}
LABEL=$P-a$A
TASKS=$(python3 -c "import json; print(json.dumps([t for t in open('tasks.txt').read().split() if t != 'qemu-startup']))")
export LOCAL_KEY=none
mkdir -p logs
uv run vf-eval primeintellect/terminal-bench-2 -m "$LABEL" \
  --client.base-url http://127.0.0.1:$PORT/v1 --client.api-key-var LOCAL_KEY \
  --env.agent.harness.id pi --env.agent.runtime.type prime --env.taskset.tasks "$TASKS" \
  --env.agent.max-turns 60 --env.agent.timeout.rollout ${TIMEOUT:-3600} \
  -n 19 -r 1 -c "$CONC" --no-push --no-rich > "logs/$LABEL-pi.log" 2>&1
grep -E "rollout done" "logs/$LABEL-pi.log" | sed -E 's/.*task=([0-9]+) reward=([0-9.]+) turns=([0-9]+) stop=([a-z_]+).*/\1 \2 \3 \4/'
