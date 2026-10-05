#!/bin/bash
# Serve an R1s build with TensorFold as a Hugging Face Job (a10g-large: 24 GB Ampere like the 3090), the way
# serve_r1s_pod.sh does on a Prime pod, and run TB2 attempts against it from here through the jobs proxy
# (https://<job>--8080.hf.jobs, the HF token as the API key; the proxy strips it before TensorFold sees the request).
#
#   ./hf_serve.sh submit <bucket path of the build, e.g. sd-out/round2-1004b/r1s-a> <served name>   # prints the job id
#   ATTEMPTS="1 2 3" HARNESS=pi ./hf_serve.sh eval <job id> <label>     # waits for the server, runs, cancels the job
#   (inside the job) bash /code/hf_serve.sh entry
#
# The code (TensorFold, its env lock, serve_kernels.py, this script) goes to the private bucket sd-in/tensorfold and
# is mounted read-only at /code; the build is mounted read-only at /model and copied to local disk before loading.
# Each attempt is run_eval.sh tb2 (tasks.txt, 60 turns, 1-hour rollouts, 4 at a time) labelled <label>-a<N>.
set -euo pipefail
HF="uvx --from huggingface_hub hf"
case ${1:-} in
submit)
    MODEL=$2 NAME=$3
    cd "$(dirname "$0")/../.."
    NS=$($HF auth whoami --json | python3 -c "import json,sys; print(json.load(sys.stdin)['user'])")
    S=/tmp/tf-stage
    rm -rf $S && mkdir -p $S/at/01-inference/tools $S/at/01-inference/envs/tensorfold $S/at/01-inference/speed-hillclimb/tensorfold
    rsync -a --exclude .git --exclude __pycache__ --exclude .venv 01-inference/tools/TensorFold $S/at/01-inference/tools/
    cp 01-inference/envs/tensorfold/pyproject.toml 01-inference/envs/tensorfold/uv.lock $S/at/01-inference/envs/tensorfold/
    cp 01-inference/speed-hillclimb/tensorfold/serve_kernels.py $S/at/01-inference/speed-hillclimb/tensorfold/
    cp 06-agents/harness-evals/hf_serve.sh $S/
    $HF buckets create "$NS/sd-in" --private --exist-ok > /dev/null
    $HF buckets sync $S "hf://buckets/$NS/sd-in/tensorfold" --delete > /dev/null
    $HF jobs run --detach -q --flavor ${FLAVOR:-a10g-large} --timeout ${TIMEOUT:-10h} --expose 8080 --name "serve-$NAME" \
        -e NAME=$NAME -e CONTEXT=${CONTEXT:-131072} \
        -v "hf://buckets/$NS/sd-in/tensorfold:/code:ro" -v "hf://buckets/$NS/$MODEL:/model:ro" \
        nvidia/cuda:13.0.1-devel-ubuntu22.04 bash /code/hf_serve.sh entry
    ;;
eval)
    J=$2 LABEL=$3
    cd "$(dirname "$0")"
    export HF_TOKEN=$(cat ${HF_HOME:-$HOME/.cache/huggingface}/token)
    U=https://$J--8080.hf.jobs/v1
    log() { echo "$(date +%T) $*" >> logs/hf-series.log; }
    up() { curl -sf -m 10 -H "Authorization: Bearer $HF_TOKEN" $U/models > /dev/null; }
    for i in $(seq 180); do up && break; sleep 20; done
    if ! up; then log "$LABEL: server on job $J never came up"
    else
        for a in ${ATTEMPTS:-1 2 3}; do
            log "$LABEL-a$a ${HARNESS:-pi} start (job $J)"
            BASE_URL=$U KEY_VAR=HF_TOKEN TIMEOUT=3600 ./run_eval.sh tb2 ${HARNESS:-pi} "$LABEL-a$a" 4 > /dev/null 2>&1
            log "$LABEL-a$a ${HARNESS:-pi} done (exit $?): $(grep -c 'reward=1' logs/tb2-$LABEL-a$a-${HARNESS:-pi}.log) solved"
            up || { log "$LABEL: server gone after attempt $a; stopping"; break; }
        done
    fi
    [ -n "${KEEP:-}" ] || { $HF jobs cancel $J > /dev/null 2>&1; log "$LABEL: cancelled job $J"
        echo "$(date -Iseconds) TERMINATE hfjob $J ($LABEL series done)" >> /tmp/prime-spend.log; }
    ;;
entry)
    say() { echo "$(date +%T) $*"; }
    say "job start: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader), $(nproc) CPUs, $(free -g | awk '/Mem/ {print $2}') GB RAM"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq > /tmp/apt.log 2>&1 && apt-get install -y -qq curl ca-certificates build-essential git >> /tmp/apt.log 2>&1 \
        || { say "apt failed"; tail -20 /tmp/apt.log; exit 1; }
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
    export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH CUDA_HOME=/usr/local/cuda
    cp -r /code/at $HOME/at
    mkdir -p $HOME/build && cp -r /model/. $HOME/build/
    say "build copied: $(du -sh $HOME/build | cut -f1), $(ls $HOME/build | wc -l) files"
    cd $HOME/at/01-inference/envs/tensorfold
    uv sync -q
    say "serving $NAME"
    TENSORFOLD_NO_UPDATE_CHECK=1 TENSORFOLD_CUDA_RESERVE_GIB=0.3 TF_GPU_ONLY_BUDGET=1 TF_RANDOM_SEED=1 TF_PASS_UNKNOWN_TOOLS=1 \
        exec uv run python ../../speed-hillclimb/tensorfold/serve_kernels.py new serve $HOME/build --host 0.0.0.0 \
        --name $NAME --context $CONTEXT --port 8080 --prompt-cache-gib 0 --swap-gib 8
    ;;
*) echo "usage: hf_serve.sh submit <bucket path> <name> | eval <job id> <label> | entry"; exit 1 ;;
esac
