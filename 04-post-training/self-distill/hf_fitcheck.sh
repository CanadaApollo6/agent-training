#!/bin/bash
# fit_check.py on one Hugging Face Job h200: vLLM 0.30 scores data/fitcheck.parquet under base Ornith (from the Hub),
# R1s-SD's bf16 (bucket sd-out/ornith-sd-bf16) and both round-2 arms (sd-out/round2-1004b/ornith-{a,b}-bf16), one model
# after another. Results land in the bucket at sd-out/fitcheck/<name>.jsonl.
#
#   ./hf_fitcheck.sh submit          # prints the job id
#   ./hf_fitcheck.sh fetch           # results to results/fitcheck/
set -euo pipefail
HF="uvx --from huggingface_hub hf"
NS() { $HF auth whoami --json | python3 -c "import json,sys; print(json.load(sys.stdin)['user'])"; }
case ${1:-} in
submit)
    cd "$(dirname "$0")"
    S=/tmp/fitcheck-stage && rm -rf $S && mkdir -p $S
    cp fit_check.py hf_fitcheck.sh data/fitcheck.parquet $S/
    U=$(NS)
    $HF buckets sync $S "hf://buckets/$U/sd-in/fitcheck" --delete > /dev/null
    $HF jobs run --detach -q --flavor ${FLAVOR:-h200} --timeout ${TIMEOUT:-3h} --name sd-fitcheck \
        -v "hf://buckets/$U/sd-in/fitcheck:/inputs:ro" -v "hf://buckets/$U/sd-out:/out" \
        nvidia/cuda:13.0.1-devel-ubuntu22.04 bash /inputs/hf_fitcheck.sh entry
    ;;
fetch)
    cd "$(dirname "$0")"
    $HF buckets sync "hf://buckets/$(NS)/sd-out/fitcheck" results/fitcheck
    ;;
entry)
    say() { echo "$(date +%T) $*"; }
    say "job start: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    export DEBIAN_FRONTEND=noninteractive PATH=$HOME/.local/bin:$PATH
    apt-get update -qq > /tmp/apt.log 2>&1 && apt-get install -y -qq curl ca-certificates build-essential >> /tmp/apt.log 2>&1
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
    uv venv -q ~/vllm --python 3.12 && VIRTUAL_ENV=~/vllm uv pip install -q "vllm==0.30.0" pyarrow "huggingface_hub[hf_xet]"
    say "vllm installed"
    mkdir -p /out/fitcheck /workspace
    ( ~/vllm/bin/hf download ornith-ai/Ornith-1.5-35B-A3B --local-dir /workspace/base > /tmp/dl.log 2>&1 ) &
    dl=$!
    for m in base:/workspace/base sd:/out/ornith-sd-bf16 a:/out/round2-1004b/ornith-a-bf16 b:/out/round2-1004b/ornith-b-bf16; do
        name=${m%%:*} src=${m#*:}
        if [ $name = base ]; then wait $dl; dir=$src
        else rm -rf /workspace/m && cp -r $src /workspace/m && dir=/workspace/m; fi
        say "scoring $name ($(du -sh $dir | cut -f1))"
        VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CONFIG_ROOT=$HOME/.vllm ~/vllm/bin/python /inputs/fit_check.py score \
            --model $dir --name $name --data /inputs/fitcheck.parquet --out /workspace/$name.jsonl > /tmp/score-$name.log 2>&1 \
            && cp /workspace/$name.jsonl /out/fitcheck/ && say "$name done" \
            || { say "$name failed"; tail -30 /tmp/score-$name.log; }
    done
    say "job end"
    ;;
*) echo "usage: hf_fitcheck.sh submit | fetch | entry"; exit 1 ;;
esac
