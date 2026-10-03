#!/bin/bash
# One harness x one served model on a Harbor taskset (Terminal-Bench 2 or DeepSWE), sandboxes on Prime.
#
#   ./run_eval.sh <tb2|tb2full|deepswe|tb4> <pi|prime_agent|claude_code> <label> [concurrency]
# Env: PORT (default 8000), TURNS, TIMEOUT seconds per rollout, ROLLOUTS (default 1), TASKS_FILE (overrides the
# suite's task list, e.g. tb2_budget_pi.txt for the runs that hit a limit). BASE_URL and KEY_VAR (the name of the env var
# holding the key) point at a hosted API instead of the local server; the label is then the API's model id.
# EXTRA: more vf-eval flags, e.g. "--sampling.max-tokens 32768 --sampling.reasoning-effort high".
# tb2: tasks.txt (20), 60 turns, 60 min. tb2full: all 89 tasks (tb2_full_tasks.txt), same caps. deepswe: deepswe_tasks.txt (10: seed 0 of the seed-0 25 of 113; Prime allows 10 private images), 100 turns, 90 min, run from
# deepswe-env/ (newer verifiers, and our own copies of the task images; see deepswe-env/README.md). tb4: tb4_tasks.txt
# (20, seed 0 of the 52 single-container CPU tasks in Terminal-Bench 4.0.0), 100 turns, 90 min, also from deepswe-env/.
# claude_code exists only in deepswe-env's verifiers, so tb2 and tb2full with it run there too, on the same taskset
# (deepswe-env/terminal-bench-2).
set -euo pipefail
cd "$(dirname "$0")"
SUITE=$1; HARNESS=$2; LABEL=$3; CONC=${4:-4}
case $SUITE in
  tb2) ENV=primeintellect/terminal-bench-2; FILE=tasks.txt; TURNS=${TURNS:-60}; TIMEOUT=${TIMEOUT:-3600}; PROJECT=. ;;
  tb2full) ENV=primeintellect/terminal-bench-2; FILE=tb2_full_tasks.txt; TURNS=${TURNS:-60}; TIMEOUT=${TIMEOUT:-3600}
    PROJECT=. ;;
  deepswe) ENV=deep-swe-upstream; FILE=deepswe_tasks.txt; TURNS=${TURNS:-100}; TIMEOUT=${TIMEOUT:-5400}
    PROJECT=deepswe-env ;;
  tb4) ENV=terminal-bench-4; FILE=tb4_tasks.txt; TURNS=${TURNS:-100}; TIMEOUT=${TIMEOUT:-5400}; PROJECT=deepswe-env ;;
  *) echo "suite: tb2, tb2full, deepswe or tb4"; exit 1 ;;
esac
if [ "$HARNESS" = claude_code ] && [ "$PROJECT" = . ]; then ENV=terminal-bench-2; PROJECT=deepswe-env; fi
FILE=${TASKS_FILE:-$FILE}
TASKS=$(python3 -c "import json; print(json.dumps(open('$FILE').read().split()))")
N=$(python3 -c "print(len(open('$FILE').read().split()))")
export LOCAL_KEY=none
mkdir -p logs
uv run --project "$PROJECT" vf-eval "$ENV" -m "$LABEL" \
  --client.base-url ${BASE_URL:-http://127.0.0.1:${PORT:-8000}/v1} --client.api-key-var ${KEY_VAR:-LOCAL_KEY} \
  --env.agent.harness.id "$HARNESS" --env.agent.runtime.type prime --env.taskset.tasks "$TASKS" \
  --env.agent.max-turns "$TURNS" --env.agent.timeout.rollout "$TIMEOUT" \
  -n "$N" -r ${ROLLOUTS:-1} -c "$CONC" --no-push --no-rich ${EXTRA:-} > "logs/$SUITE-${LABEL//\//_}-$HARNESS.log" 2>&1
grep -E "rollout done" "logs/$SUITE-${LABEL//\//_}-$HARNESS.log" | sed -E 's/.*task=([0-9]+) reward=([0-9.]+) turns=([0-9]+) stop=([A-Za-z_]+).*/\1 \2 \3 \4/'
