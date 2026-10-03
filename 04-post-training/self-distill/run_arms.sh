#!/bin/bash
# Round 2 on one 8-GPU pod: arm A (own + raw teacher traces) and arm B (own + teacher traces with Ornith's rewritten
# reasoning) trained back to back with the same settings, then both R1s rebuilds side by side. Staged by stage.sh with
# DATA="data/train-a data/train-b".
#
#   W=/workspace STEPS_A=70 STEPS_B=66 bash run_arms.sh
set -euo pipefail
cd "$(dirname "$0")"
: "${STEPS_A:?} ${STEPS_B:?}"
bash pod_train.sh setup
ARM=a bash pod_train.sh train --max-steps "$STEPS_A"
ARM=a bash pod_train.sh export
ARM=b bash pod_train.sh train --max-steps "$STEPS_B"
ARM=b bash pod_train.sh export
GPUS="0 1" ARM=a bash pod_train.sh rebuild &
GPUS="2 3" ARM=b bash pod_train.sh rebuild &
wait
echo "$(date +%T) all done" >> pod.log
