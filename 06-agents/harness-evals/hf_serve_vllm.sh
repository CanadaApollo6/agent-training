#!/bin/bash
# Serve a Hugging Face model with vLLM as a Hugging Face Job (default: Qwen/Qwen3.8-27B in bf16 on one H200, $5/h) and
# run prime_agent attempts on the held-out 20 against it from here through the jobs proxy
# (https://<job>--8080.hf.jobs, the HF token as the API key; the proxy strips it before vLLM sees the request).
#
#   ./hf_serve_vllm.sh submit <label prefix>       # prints the job id; the labels <prefix>-a1..a3 are served names
#   ./hf_serve_vllm.sh eval <job id> <label prefix> # waits for the server, runs a1..a3 side by side, cancels the job
#   (inside the job) bash /code/hf_serve_vllm.sh entry
#
# Qwen3.8's template defaults to reasoning_effort xhigh, which adds "Reasoning effort is set to xhigh" to the system
# prompt and made it overthink in the math probe (01-inference/qwen38-27b README). EFFORT (default medium) is set
# server-side with --default-chat-template-kwargs, so every request gets it whatever the harness sends.
# Same caps as R1s-SD's c32 runs: 131K window, 32K per-call reply cap, 60 turns, 1-hour rollouts.
set -euo pipefail
HF="uvx --from huggingface_hub hf"
case ${1:-} in
submit)
    P=$2
    cd "$(dirname "$0")"
    NS=$($HF auth whoami --json | python3 -c "import json,sys; print(json.load(sys.stdin)['user'])")
    S=/tmp/vllm-stage; rm -rf $S && mkdir -p $S && cp hf_serve_vllm.sh $S/
    $HF buckets create "$NS/sd-in" --private --exist-ok > /dev/null
    $HF buckets sync $S "hf://buckets/$NS/sd-in/vllm-serve" --delete > /dev/null
    $HF jobs run --detach -q --flavor ${FLAVOR:-h200} --timeout ${TIMEOUT:-5h} --expose 8080 --name "serve-$P" \
        -e REPO=${REPO:-Qwen/Qwen3.8-27B} -e NAMES="$P-a1 $P-a2 $P-a3" -e EFFORT=${EFFORT:-medium} \
        -e CONTEXT=${CONTEXT:-131072} -v "hf://buckets/$NS/sd-in/vllm-serve:/code:ro" \
        nvidia/cuda:13.0.1-devel-ubuntu22.04 bash /code/hf_serve_vllm.sh entry
    ;;
eval)
    J=$2 P=$3
    cd "$(dirname "$0")"
    export HF_TOKEN=$(cat ${HF_HOME:-$HOME/.cache/huggingface}/token)
    U=https://$J--8080.hf.jobs/v1
    log() { echo "$(date +%T) $*" >> logs/hf-series.log; }
    up() { curl -sf -m 10 -H "Authorization: Bearer $HF_TOKEN" $U/models > /dev/null; }
    for i in $(seq 180); do up && break; sleep 20; done
    if ! up; then log "$P: vLLM on job $J never came up"
    else
        for a in 1 2 3; do
            log "$P-a$a prime_agent start (job $J)"
            ( BASE_URL=$U KEY_VAR=HF_TOKEN TIMEOUT=3600 EXTRA="--sampling.max-tokens 32768" \
                ./run_eval.sh tb2 prime_agent "$P-a$a" ${CONC:-8} > /dev/null 2>&1
              log "$P-a$a prime_agent done (exit $?): $(grep -c 'reward=1' logs/tb2-$P-a$a-prime_agent.log) solved" ) &
            sleep 30
        done
        wait
    fi
    $HF jobs cancel $J > /dev/null 2>&1; log "$P: cancelled job $J"
    echo "$(date -Iseconds) TERMINATE hfjob $J ($P series done)" >> /tmp/prime-spend.log
    ;;
entry)
    say() { echo "$(date +%T) $*"; }
    say "job start: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader), $(nproc) CPUs"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq > /tmp/apt.log 2>&1 && apt-get install -y -qq curl ca-certificates build-essential git >> /tmp/apt.log 2>&1 \
        || { say "apt failed"; tail -20 /tmp/apt.log; exit 1; }
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
    export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH CUDA_HOME=/usr/local/cuda
    uv venv -q ~/vllm --python 3.12
    VIRTUAL_ENV=~/vllm uv pip install -q vllm "transformers>=5.8.0" "huggingface_hub[hf_xet]" 2>&1 | tail -5
    say "vllm $(~/vllm/bin/python -c 'import vllm; print(vllm.__version__)'), transformers $(~/vllm/bin/python -c 'import transformers; print(transformers.__version__)')"
    ~/vllm/bin/hf download $REPO --local-dir ~/model > /tmp/download.log 2>&1 || { say "download failed"; tail /tmp/download.log; exit 1; }
    say "model downloaded: $(du -sh ~/model | cut -f1)"
    say "serving $REPO as: $NAMES (reasoning_effort $EFFORT, context $CONTEXT)"
    exec ~/vllm/bin/vllm serve ~/model --served-model-name $NAMES --host 0.0.0.0 --port 8080 \
        --max-model-len $CONTEXT --gpu-memory-utilization 0.92 --enable-prefix-caching \
        --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
        --default-chat-template-kwargs "{\"reasoning_effort\": \"$EFFORT\"}"
    ;;
*) echo "usage: hf_serve_vllm.sh submit <label prefix> | eval <job id> <label prefix> | entry"; exit 1 ;;
esac
