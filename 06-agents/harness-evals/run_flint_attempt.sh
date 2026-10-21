#!/bin/bash
# One attempt of the Flint Terminal-Bench 2 pilot: pi on the 19 tasks of tasks.txt (qemu-startup dropped), against
# Flint BF16 on SGLang (serve_flint_pod.sh), tunnelled to 127.0.0.1:<port>. Sandboxes on Prime VMs. Same caps as the
# R1s and LFM2.5 pilots: 60 turns, 60 minutes. pi sends no sampling, so the server's generation_config defaults apply
# (temperature 1.0, top_p 0.95, top_k 20). SGLang batches, so one server takes all attempts at once.
#
#   ./run_flint_attempt.sh <attempt number> <port> [concurrency]
set -euo pipefail
cd "$(dirname "$0")"
A=$1; PORT=$2; CONC=${3:-7}
LABEL=flint-bf16-a$A
TASKS=$(python3 -c "import json; print(json.dumps([t for t in open('tasks.txt').read().split() if t != 'qemu-startup']))")
export LOCAL_KEY=none
mkdir -p logs
uv run vf-eval primeintellect/terminal-bench-2 -m "$LABEL" \
  --client.base-url http://127.0.0.1:$PORT/v1 --client.api-key-var LOCAL_KEY \
  --env.agent.harness.id pi --env.agent.runtime.type prime --env.taskset.tasks "$TASKS" \
  --env.agent.max-turns 60 --env.agent.timeout.rollout ${TIMEOUT:-3600} \
  -n 19 -r 1 -c "$CONC" --no-push --no-rich > "logs/$LABEL-pi.log" 2>&1
grep -E "rollout done" "logs/$LABEL-pi.log" | sed -E 's/.*task=([0-9]+) reward=([0-9.]+) turns=([0-9]+) stop=([a-z_]+).*/\1 \2 \3 \4/'
