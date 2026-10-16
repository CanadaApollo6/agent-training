#!/bin/bash
# router_stats.py as a Hugging Face Job on a100x8: the BF16 model (360 GB) fits across 8x 80 GB. Token windows and
# the script go up to a private bucket mounted read-only at /inputs; results come back through a second private bucket
# mounted at /out, copied every 2 minutes. Same pattern as 04-post-training/self-distill/hf_job.sh.
#
#   ./hf_job.sh submit <data dir>     # upload, submit; prints the run name
#   ./hf_job.sh fetch <run>           # download hf://buckets/<user>/flash-out/<run> to results/<run>
#   (inside the job) bash /inputs/hf_job.sh entry
set -euo pipefail
HF="uvx --from huggingface_hub hf"
case ${1:-} in
submit)
    cd "$(dirname "$0")"
    USER_NS=$($HF auth whoami --json | python3 -c "import json,sys; print(json.load(sys.stdin)['user'])")
    RUN=${RUN:-experts-$(date +%m%d-%H%M)}
    S=/tmp/flash-stage-$RUN
    mkdir -p $S && cp "$2"/*.npz "$2"/stats.json router_stats.py hf_job.sh $S/
    $HF buckets create "$USER_NS/flash-in" --private --exist-ok > /dev/null
    $HF buckets create "$USER_NS/flash-out" --private --exist-ok > /dev/null
    $HF buckets sync $S "hf://buckets/$USER_NS/flash-in/$RUN" > /dev/null
    echo "HF_TOKEN=$(cat ${HF_HOME:-$HOME/.cache/huggingface}/token)" | $HF jobs run --detach -q --secrets-file - \
        --flavor ${FLAVOR:-a100x8} --timeout ${TIMEOUT:-3h} --name "flash-$RUN" \
        -e RUN=$RUN -e NS=$USER_NS -e LIMIT_CALIB=${LIMIT_CALIB:-} -e LIMIT_HELDOUT=${LIMIT_HELDOUT:-} -e LIMIT_CONTROL=${LIMIT_CONTROL:-} \
        -v "hf://buckets/$USER_NS/flash-in/$RUN:/inputs:ro" -v "hf://buckets/$USER_NS/flash-out:/out" \
        nvidia/cuda:13.0.1-devel-ubuntu22.04 bash /inputs/hf_job.sh entry
    echo "run $RUN"
    ;;
fetch)
    USER_NS=$($HF auth whoami --json | python3 -c "import json,sys; print(json.load(sys.stdin)['user'])")
    $HF buckets sync "hf://buckets/$USER_NS/flash-out/$2" "$(dirname "$0")/results/$2"
    ;;
entry)
    set +e
    W=/workspace OUT=/out/$RUN
    mkdir -p $W/out $OUT
    say() { echo "$(date +%T) $*" | tee -a $OUT/job.log; }
    say "job start: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq > /tmp/apt.log 2>&1 && apt-get install -y -qq curl ca-certificates python3 build-essential \
        >> /tmp/apt.log 2>&1 || { say "apt failed"; tail -20 /tmp/apt.log; exit 1; }
    command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
    export PATH=$HOME/.local/bin:$PATH
    cp -r /inputs $W/in
    t0=$(date +%s)
    HF_XET_HIGH_PERFORMANCE=1 uvx --from huggingface_hub hf download Qwen/Qwen3.8-Flash-Next --local-dir $W/model > /tmp/dl.log 2>&1 \
        || { say "download failed"; tail -20 /tmp/dl.log | tee -a $OUT/job.log; exit 1; }
    say "model downloaded: $(du -sh $W/model | cut -f1) in $(( $(date +%s) - t0 ))s"
    ( while sleep 120; do cp $W/out/* $OUT/ 2>/dev/null; done ) &
    cd $W/in && uv run --no-project -q --python 3.12 --with torch --with transformers==5.20.1 --with accelerate \
        --with flash-linear-attention --with numpy python -I router_stats.py $W/model $W/in $W/out \
        > $W/out/run.log 2>&1
    rc=$?
    cp $W/out/* $OUT/
    say "router_stats exit $rc"
    exit $rc
    ;;
*) echo "usage: hf_job.sh submit <data dir>|fetch <run>|entry"; exit 1 ;;
esac
