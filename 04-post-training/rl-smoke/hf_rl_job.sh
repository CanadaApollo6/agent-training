#!/bin/bash
# prime-rl smoke test as a Hugging Face Job on a100x8: Ornith-1.5-9B, GRPO, prime_agent on TB2 training tasks in Prime
# sandboxes (ornith9b-tb2-smoke.toml). The sandboxed agent reaches the model through a Prime tunnel (tunnel_check.py
# showed HF Jobs can open one). This dir goes up to a private bucket mounted read-only at /inputs; logs come back
# every 2 minutes through the rl-out bucket mounted at /out. Checkpoints stay on the job (smoke test: not uploaded).
# Secrets (HF token, Prime key) go over stdin, never on a command line.
# MODEL_SRC=<dir in the sd-out bucket> (e.g. ornith-sd-bf16) mounts sd-out read-only and copies that checkpoint to
# /workspace/<dir> before training; the config's [model] name points there.
# PI_CONTEXT_WINDOW=<tokens> patches that contextWindow into the pi harness's models.json (verifiers writes none, and
# pi then assumes 128000): pi compacts once its context passes contextWindow - 16384.
# EXPORT=1 (needs MODEL_SRC and a [ckpt] in the config) turns the run's last checkpoint into bf16 weights after training,
# measures how many weights changed (pilot/weights_moved.py) and uploads them to the sd-out bucket as <run>-bf16.
# RL_TIMEOUT=<duration> (e.g. 7h) stops training cleanly after that long, so a long run still leaves the job time to
# export its latest checkpoint before TIMEOUT kills the job.
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
    [ -n "${EXPORT:-}" ] && $HF buckets cp ../self-distill/finish_export.py "hf://buckets/$NS/rl-in/$RUN/finish_export.py" > /dev/null
    MOUNT=(); [ -n "${MODEL_SRC:-}" ] && MOUNT=(-v "hf://buckets/$NS/sd-out:/sd:ro")
    { echo "HF_TOKEN=$(cat ${HF_HOME:-$HOME/.cache/huggingface}/token)"
      echo "PRIME_API_KEY=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.prime/config.json')))['api_key'])")"; } |
    $HF jobs run --detach -q --secrets-file - --flavor ${FLAVOR:-a100x8} --timeout ${TIMEOUT:-270m} --name "rl-$RUN" \
        -e RUN=$RUN -e NS=$NS -e CONFIG=${CONFIG:-ornith9b-tb2-smoke.toml} -e PRIME_RL_COMMIT=$PRIME_RL_COMMIT \
        -e MODEL_SRC=${MODEL_SRC:-} -e PI_CONTEXT_WINDOW=${PI_CONTEXT_WINDOW:-} -e EXPORT=${EXPORT:-} -e RL_TIMEOUT=${RL_TIMEOUT:-} "${MOUNT[@]}" \
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
    if [ -n "${PI_CONTEXT_WINDOW:-}" ]; then
        PI_PY=deps/verifiers/verifiers/v1/harnesses/pi/harness.py
        sed -i "s/^\( *\)\"input\": \[\"text\", \"image\"\],/&\n\1\"contextWindow\": $PI_CONTEXT_WINDOW,/" $PI_PY
        grep -q "\"contextWindow\": $PI_CONTEXT_WINDOW," $PI_PY && say "pi contextWindow patched: $PI_CONTEXT_WINDOW" \
            || { say "pi contextWindow patch failed"; exit 1; }
    fi
    uv sync --extra gpu --extra flash-attn --extra kernels --extra disagg --package prime-rl --package terminal-bench-2 > /tmp/sync.log 2>&1 \
        || { say "uv sync failed"; tail -40 /tmp/sync.log >> $OUT/job.log; exit 1; }
    say "uv sync done; vllm-router: $(ls .venv/bin/vllm-router 2>/dev/null || echo MISSING)"
    if [ -n "${MODEL_SRC:-}" ]; then
        cp -r /sd/$MODEL_SRC /workspace/$MODEL_SRC || { say "model copy failed"; exit 1; }
        say "model copied: $(du -sh /workspace/$MODEL_SRC | cut -f1)"
    fi
    ( while sleep 120; do
        rsync -a --exclude checkpoints --exclude weights --exclude broadcasts --exclude '*.safetensors' --exclude '*.pt' \
            --exclude '*.distcp' outputs/ $OUT/outputs/ 2>/dev/null
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader > $OUT/gpu.txt 2>/dev/null
      done ) &
    say "rl start"
    RLT=(); [ -n "${RL_TIMEOUT:-}" ] && RLT=(timeout -k 300 $RL_TIMEOUT)   # SIGTERM: rl stops all its processes
    "${RLT[@]}" uv run rl @ /inputs/$CONFIG --run.name $RUN --output-dir outputs > $OUT/rl.stdout 2>&1
    rc=$?
    say "rl exit $rc"
    rsync -a --exclude checkpoints --exclude weights --exclude broadcasts --exclude '*.safetensors' --exclude '*.pt' \
        --exclude '*.distcp' outputs/ $OUT/outputs/
    # newest complete checkpoint (DCP writes .metadata last; a save cut off by RL_TIMEOUT has none)
    ckpt=$(ls -d outputs/$RUN/checkpoints/step_* 2>/dev/null | sort -t_ -k2 -n | while read d; do
        [ -f $d/trainer/.metadata ] && echo $d; done | tail -1)
    if [ -n "${EXPORT:-}" ] && [ -n "$ckpt" ]; then
        # the run's last checkpoint -> bf16 HF weights, MTP head and assets put back from the starting model (checks
        # the frozen vision tower and routers), share of weights changed, then up to sd-out/<run>-bf16
        EXP=/workspace/$RUN-bf16
        say "exporting $ckpt"
        uv run torchrun --nproc-per-node 8 tools/convert_dcp_to_bf16.py $ckpt $EXP > $OUT/export.log 2>&1 \
            || [ -f $EXP/model.safetensors.index.json ] || { say "export failed"; exit 1; }
        uv run python /inputs/finish_export.py --orig /workspace/$MODEL_SRC --export $EXP > $OUT/finish_export.log 2>&1 \
            || { say "finish_export failed"; exit 1; }
        uv run python /inputs/pilot/weights_moved.py /workspace/$MODEL_SRC $EXP --out $OUT/weights_moved.json \
            > $OUT/weights_moved.txt 2>&1
        say "weights changed: $(grep '^all' $OUT/weights_moved.txt)"
        $HF buckets sync $EXP "hf://buckets/$NS/sd-out/$RUN-bf16" > $OUT/upload.log 2>&1 \
            && say "uploaded $(du -sh $EXP | cut -f1) to sd-out/$RUN-bf16" || say "upload failed"
    fi
    say "checkpoints on the job (not uploaded): $(du -sh outputs 2>/dev/null | cut -f1)"
    say "job end"
    exit $rc
    ;;
*) echo "usage: hf_rl_job.sh submit|entry"; exit 1 ;;
esac
