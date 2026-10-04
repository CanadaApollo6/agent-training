#!/bin/bash
# run_arms.sh as a Hugging Face Job on a100x8 (when Prime has no 8-GPU node). The staged inputs go up to a private
# bucket mounted read-only at /inputs. Results come back through a second private bucket mounted at /out: logs every
# 5 minutes through the mount, then r1s-{a,b} and the tuned bf16 copies uploaded with hf buckets sync (the job gets
# the local HF token as a secret, read from the token file over stdin, never on a command line).
# DRY=1 checks the path on cpu-basic: apt, inputs, the log mount, a test upload, no training.
#
#   STEPS_A=70 STEPS_B=68 ./hf_job.sh submit        # stage, upload, submit; prints the job id
#   ./hf_job.sh fetch <run>                         # download hf://buckets/<user>/sd-out/<run> to ~/models/ornith
#   (inside the job) bash /inputs/hf_job.sh entry
set -euo pipefail
HF="uvx --from huggingface_hub hf"
case ${1:-} in
submit)
    : "${STEPS_A:?} ${STEPS_B:?}"
    cd "$(dirname "$0")"
    USER_NS=$($HF auth whoami --json | python3 -c "import json,sys; print(json.load(sys.stdin)['user'])")
    RUN=${RUN:-round2-$(date +%m%d-%H%M)}
    S=/tmp/sd-stage-$RUN
    DATA=${DATA:-"data/train-a data/train-b"} ./stage.sh local "$S"
    cp hf_job.sh "$S/sd/"
    $HF buckets create "$USER_NS/sd-in" --private --exist-ok > /dev/null
    $HF buckets create "$USER_NS/sd-out" --private --exist-ok > /dev/null
    $HF buckets sync "$S/sd" "hf://buckets/$USER_NS/sd-in/$RUN" > /dev/null
    [ "${DRY:-0}" = 1 ] && FLAVOR=cpu-basic TIMEOUT=15m
    echo "HF_TOKEN=$(cat ${HF_HOME:-$HOME/.cache/huggingface}/token)" | $HF jobs run --detach -q --secrets-file - \
        --flavor ${FLAVOR:-a100x8} --timeout ${TIMEOUT:-5h} --name "sd-$RUN" \
        -e RUN=$RUN -e NS=$USER_NS -e STEPS_A=$STEPS_A -e STEPS_B=$STEPS_B -e DRY=${DRY:-0} \
        -v "hf://buckets/$USER_NS/sd-in/$RUN:/inputs:ro" -v "hf://buckets/$USER_NS/sd-out:/out" \
        nvidia/cuda:13.0.1-devel-ubuntu22.04 bash /inputs/hf_job.sh entry
    echo "run $RUN"
    ;;
fetch)
    USER_NS=$($HF auth whoami --json | python3 -c "import json,sys; print(json.load(sys.stdin)['user'])")
    $HF buckets sync "hf://buckets/$USER_NS/sd-out/$2" "$HOME/models/ornith/$2"
    ;;
entry)
    set +e
    W=/workspace OUT=/out/$RUN
    mkdir -p $W $OUT/logs
    say() { echo "$(date +%T) $*" | tee -a $OUT/logs/job.log; }
    say "job start: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c), /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $2}')"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq > /tmp/apt.log 2>&1 && apt-get install -y -qq git curl ca-certificates rsync build-essential \
        python3 >> /tmp/apt.log 2>&1 || { say "apt failed"; tail -20 /tmp/apt.log; exit 1; }
    # containers often get a small /dev/shm; NCCL then falls back to its socket path for the shared-memory transport
    shm_kb=$(df -k /dev/shm | tail -1 | awk '{print $2}')
    [ "$shm_kb" -lt 8000000 ] && export NCCL_SHM_DISABLE=1 && say "small /dev/shm: NCCL_SHM_DISABLE=1"
    cp -r /inputs $W/sd
    command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
    export PATH=$HOME/.local/bin:$PATH
    up() { uvx --from huggingface_hub hf buckets sync "$1" "hf://buckets/$NS/sd-out/$RUN/$(basename $1)" > /tmp/up.log 2>&1 \
        && say "uploaded $(basename $1) $(du -sh $1 | cut -f1)" || { say "upload of $1 failed"; tail -5 /tmp/up.log; }; }
    if [ "${DRY:-0}" = 1 ]; then
        say "inputs: $(du -sh $W/sd | cut -f1), $(ls $W/sd | tr '\n' ' ')"
        mkdir -p $W/dry && head -c 50000000 /dev/urandom > $W/dry/blob && up $W/dry
        say "dry run done"; exit 0
    fi
    ( while sleep 300; do cp $W/sd/*.log $OUT/logs/ 2>/dev/null; done ) &
    W=$W STEPS_A=$STEPS_A STEPS_B=$STEPS_B bash $W/sd/run_arms.sh
    rc=$?
    say "run_arms exit $rc"
    cp $W/sd/*.log $OUT/logs/
    for a in a b; do [ -f $W/r1s-$a/config.json ] && up $W/r1s-$a; done
    for a in a b; do [ -f $W/ornith-$a-bf16/model.safetensors.index.json ] && up $W/ornith-$a-bf16; done
    say "job end"
    exit $rc
    ;;
*) echo "usage: hf_job.sh submit|fetch <run>|entry"; exit 1 ;;
esac
