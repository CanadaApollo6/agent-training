#!/bin/bash
# One harness x one served model on a Harbor taskset (Terminal-Bench 2 or DeepSWE), sandboxes on Prime.
#
#   ./run_eval.sh <tb2|deepswe> <pi|prime_agent> <label> [concurrency]
# Env: PORT (default 8000), TURNS, TIMEOUT seconds per rollout, ROLLOUTS (default 1).
# tb2: tasks.txt (20), 60 turns, 60 min. deepswe: deepswe_tasks.txt (25, seed 0 of 113), 100 turns, 90 min, run from
# deepswe-env/ (newer verifiers, and our own copies of the task images; see deepswe-env/README.md).
set -euo pipefail
cd "$(dirname "$0")"
SUITE=$1; HARNESS=$2; LABEL=$3; CONC=${4:-4}
case $SUITE in
  tb2) ENV=primeintellect/terminal-bench-2; FILE=tasks.txt; TURNS=${TURNS:-60}; TIMEOUT=${TIMEOUT:-3600}; PROJECT=. ;;
  deepswe) ENV=deep-swe-upstream; FILE=deepswe_tasks.txt; TURNS=${TURNS:-100}; TIMEOUT=${TIMEOUT:-5400}
    PROJECT=deepswe-env ;;
  *) echo "suite: tb2 or deepswe"; exit 1 ;;
esac
TASKS=$(python3 -c "import json; print(json.dumps(open('$FILE').read().split()))")
N=$(python3 -c "print(len(open('$FILE').read().split()))")
export LOCAL_KEY=none
mkdir -p logs
uv run --project "$PROJECT" vf-eval "$ENV" -m "$LABEL" \
  --client.base-url http://127.0.0.1:${PORT:-8000}/v1 --client.api-key-var LOCAL_KEY \
  --env.agent.harness.id "$HARNESS" --env.agent.runtime.type prime --env.taskset.tasks "$TASKS" \
  --env.agent.max-turns "$TURNS" --env.agent.timeout.rollout "$TIMEOUT" \
  -n "$N" -r ${ROLLOUTS:-1} -c "$CONC" --no-push --no-rich > "logs/$SUITE-$LABEL-$HARNESS.log" 2>&1
grep -E "rollout done" "logs/$SUITE-$LABEL-$HARNESS.log" | sed -E 's/.*task=([0-9]+) reward=([0-9.]+) turns=([0-9]+) stop=([A-Za-z_]+).*/\1 \2 \3 \4/'
