#!/bin/bash
# Teacher bake-off: hosted models under prime_agent on the training tasks no harness has solved (datagen/never.txt, 27),
# one attempt each at the eval caps, all teachers in parallel, CONC (default 4) rollouts each. Models through Prime
# Inference; the key is read from ~/.prime/config.json. Labels are the API model ids; progress in logs/bakeoff.log.
#
#   setsid nohup ./teacher_bakeoff.sh z-ai/glm-5.3 deepseek/deepseek-v4.1-flash > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
PRIME_INF_KEY=$(jq -r .api_key ~/.prime/config.json); export PRIME_INF_KEY
log() { echo "$(date +%T) $*" >> logs/bakeoff.log; }
for M in "$@"; do
  ( log "$M start"
    BASE_URL=https://api.pinference.ai/api/v1 KEY_VAR=PRIME_INF_KEY TASKS_FILE=${TASKS_FILE:-datagen/never.txt} \
      ./run_eval.sh tb2 prime_agent "$M" ${CONC:-4} > /dev/null 2>&1
    log "$M done (exit $?)" ) &
done
wait
