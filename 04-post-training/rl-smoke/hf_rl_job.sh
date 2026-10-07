#!/bin/bash
# prime-rl smoke test as a Hugging Face Job on a100x8: Ornith-1.5-9B, GRPO, prime_agent on TB2 training tasks in Prime
# sandboxes (ornith9b-tb2-smoke.toml). The sandboxed agent reaches the model through a Prime tunnel (tunnel_check.py
# showed HF Jobs can open one). This dir goes up to a private bucket mounted read-only at /inputs; logs come back
# every 2 minutes through the rl-out bucket mounted at /out. Checkpoints stay on the job (smoke test: not uploaded).
# Secrets (HF token, Prime key) go over stdin, never on a command line.
#
#   ./hf_rl_job.sh submit          # prints the job id
#   (inside the job) bash /inputs/hf_rl_job.sh entry
set -euo pipefail
HF="uvx --from huggingface_hub hf"
PRIME_RL_COMMIT=81052082a9a2be5502b735a64a3c36092d5d096a   # pins verifiers 7ba657f2, renderers d43c4301
case ${1:-} in
submit)
    cd "$(dirname "$0")"
    NS=$($HF auth whoami --json | python3 -c "import json,sys; print(json.load(sys.stdin)['user'])")
    RUN=${RUN:-ornith9b-tb2-smoke-$(date +%m%d-%H%M)}
    $HF buckets create "$NS/rl-in" --private --exist-ok > /dev/null
    $HF buckets create "$NS/rl-out" --private --exist-ok > /dev/null
    $HF buckets sync . "hf://buckets/$NS/rl-in/$RUN" > /dev/null
    { echo "HF_TOKEN=$(cat ${HF_HOME:-$HOME/.cache/huggingface}/token)"
      echo "PRIME_API_KEY=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.prime/config.json')))['api_key'])")"; } |
    $HF jobs run --detach -q --secrets-file - --flavor ${FLAVOR:-a100x8} --timeout ${TIMEOUT:-270m} --name "rl-$RUN" \
        -e RUN=$RUN -e NS=$NS -e CONFIG=${CONFIG:-ornith9b-tb2-smoke.toml} -e PRIME_RL_COMMIT=$PRIME_RL_COMMIT \
        -v "hf://buckets/$NS/rl-in/$RUN:/inputs:ro" -v "hf://buckets/$NS/rl-out:/out" \
        nvidia/cuda:13.0.1-devel-ubuntu22.04 bash /inputs/hf_rl_job.sh entry
    echo "run $RUN"
    ;;
entry)
    set +e
    OUT=/out/$RUN; mkdir -p $OUT
    say() { echo "$(date +%T) $*" | tee -a $OUT/job.log; }
    say "job start: $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | xargs)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq > /tmp/apt.log 2>&1 && apt-get install -y -qq git curl ca-certificates build-essential rsync \
        >> /tmp/apt.log 2>&1 || { say "apt failed"; tail -20 /tmp/apt.log >> $OUT/job.log; exit 1; }
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1; export PATH=$HOME/.local/bin:$PATH
    mkdir -p /workspace && cd /workspace
    git clone -q https://github.com/PrimeIntellect-ai/prime-rl && cd prime-rl && git checkout -q $PRIME_RL_COMMIT
    # .gitmodules use git@github.com: URLs; the job has no SSH keys
    GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.https://github.com/.insteadOf GIT_CONFIG_VALUE_0=git@github.com: \
        git submodule update --init deps/verifiers deps/renderers deps/pydantic-config deps/prime-envs > /tmp/sub.log 2>&1 \
        || { say "submodules failed"; tail -20 /tmp/sub.log >> $OUT/job.log; exit 1; }
    say "cloned prime-rl $(git rev-parse --short HEAD)"
    uv sync --extra gpu --extra flash-attn --extra kernels --extra disagg --package prime-rl --package terminal-bench-2 > /tmp/sync.log 2>&1 \
        || { say "uv sync failed"; tail -40 /tmp/sync.log >> $OUT/job.log; exit 1; }
    say "uv sync done; vllm-router: $(ls .venv/bin/vllm-router 2>/dev/null || echo MISSING)"
    ( while sleep 120; do
        rsync -a --exclude checkpoints --exclude weights --exclude broadcasts --exclude '*.safetensors' --exclude '*.pt' \
            --exclude '*.distcp' outputs/ $OUT/outputs/ 2>/dev/null
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader > $OUT/gpu.txt 2>/dev/null
      done ) &
    say "rl start"
    uv run rl @ /inputs/$CONFIG --run.name $RUN --output-dir outputs > $OUT/rl.stdout 2>&1
    rc=$?
    say "rl exit $rc"
    rsync -a --exclude checkpoints --exclude weights --exclude broadcasts --exclude '*.safetensors' --exclude '*.pt' \
        --exclude '*.distcp' outputs/ $OUT/outputs/
    say "checkpoints on the job (not uploaded): $(du -sh outputs 2>/dev/null | cut -f1)"
    say "job end"
    exit $rc
    ;;
*) echo "usage: hf_rl_job.sh submit|entry"; exit 1 ;;
esac
