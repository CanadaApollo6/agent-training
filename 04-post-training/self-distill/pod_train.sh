#!/bin/bash
# Self-distillation on one 8-GPU pod: train, export, rebuild R1s from the tuned weights.
#
#   W=/workspace ARM=a bash pod_train.sh [setup|train|export|rebuild|all] [extra sft args...]
#
# ARM names a training run: data from $W/sd/data/train-$ARM, results $W/run-$ARM, $W/ornith-$ARM-bf16, $W/r1s-$ARM.
# ARM=sd (the default) is round 1's layout (data/train, run). GPUS="0 1" picks the two GPUs a rebuild uses.
#
# Expects in $W/sd (rsynced from the laptop): this directory's scripts, data/train/train.parquet, the imatrix
# (Ornith-1.5-35B-A3B-imatrix.gguf), mtp-4bit.safetensors, and repo/ holding the rebuild scripts at their repo paths
# (01-inference/speed-hillclimb/tensorfold/quantize_experts.py, 05-compression/ornith/kl/{kl_harness,export_always_on,
# make_build}.py). The result is $W/r1s-$ARM, an R1s built exactly as R1s was, from the tuned bf16.
set -euo pipefail
W=${W:-/workspace}
SD=$W/sd
STAGE=${1:-all}
shift || true
ARM=${ARM:-sd}
SFX=$([ "$ARM" = sd ] || echo "-$ARM")
read -r G0 G1 <<< "${GPUS:-0 1}"
PRIME_RL_COMMIT=1bddcf6
export PATH=$HOME/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1
log() { echo "$(date +%T) $*" | tee -a $SD/pod.log; }
# prime-rl's torch is built for CUDA 13 (driver >= 580); older drivers on datacenter GPUs run it through NVIDIA's
# forward-compatibility libraries (apt install cuda-compat-13-0)
driver=$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)   # -i 0, not | head: under pipefail head's early close SIGPIPEs nvidia-smi on 8 GPUs
if [ "$driver" -lt 580 ] && [ -d /usr/local/cuda-13.0/compat ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

setup() {
    nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader | tee -a $SD/pod.log
    [ "$driver" -ge 580 ] || [ -d /usr/local/cuda-13.0/compat ] || { log "driver $driver < 580 and no cuda-compat-13-0"; exit 1; }
    command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    log "downloading weights"
    (uvx --with hf_transfer hf download ornith-ai/Ornith-1.5-35B-A3B --local-dir $W/ornith-bf16 > $SD/dl-bf16.log 2>&1) &
    (uvx --with hf_transfer hf download ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit --local-dir $W/mlx4 > $SD/dl-mlx4.log 2>&1) &
    if [ ! -d $W/prime-rl ]; then
        git -c url."https://github.com/".insteadOf="git@github.com:" clone https://github.com/PrimeIntellect-ai/prime-rl $W/prime-rl
        cd $W/prime-rl && git checkout $PRIME_RL_COMMIT
        git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init --recursive
        git apply $SD/prime-rl-pretokenized.patch
    fi
    cd $W/prime-rl && uv sync --all-extras > $SD/uv-sync.log 2>&1
    uv run python -c "import flash_attn, torch; print('torch', torch.__version__, torch.cuda.device_count(), 'GPUs', torch.ones(1).cuda().item())" | tee -a $SD/pod.log
    wait
    cp -n $SD/mtp-4bit.safetensors $W/mlx4/
    log "setup done: $(du -sh $W/ornith-bf16 | cut -f1) bf16, $(du -sh $W/mlx4 | cut -f1) mlx4"
}

train() {
    log "training $ARM $*"
    cd $W/prime-rl
    # Qwen3.5's DeltaNet context-parallel path copies cu_seqlens to the CPU in every layer. Activation checkpointing
    # saves CUDA-to-CPU copies and replays them in the backward recompute, and there the replay hands an int tensor to
    # a float op ("Autograd not support dtype: Int"). Recomputing the copy instead is correct and cheap.
    export SD_RECOMPUTE_CPU_COPIES=1
    uv run sft @ $SD/sft.toml --model.name $W/ornith-bf16 --model.conversion-dir $W/ornith-prime \
        --data.name $SD/data/train$SFX --output-dir $W/run$SFX "$@" 2>&1 | tee $SD/train-$ARM.log
    log "training $ARM done"
}

export_weights() {
    cd $W/prime-rl
    ckpt=$(ls -d $W/run$SFX/*/checkpoints/step_* | sort -t_ -k2 -n | tail -1)
    log "exporting $ARM $ckpt"
    # the converter writes all the weights, then dies saving assets (the text-only FSDP model has no generation_config);
    # finish_export.py copies the original's assets anyway, so only a missing index is fatal
    uv run torchrun --nproc-per-node 8 tools/convert_dcp_to_bf16.py $ckpt $W/ornith-$ARM-bf16 \
        > $SD/export-$ARM.log 2>&1 || [ -f $W/ornith-$ARM-bf16/model.safetensors.index.json ]
    uv run python $SD/finish_export.py --orig $W/ornith-bf16 --export $W/ornith-$ARM-bf16 | tee $SD/finish-export-$ARM.log
    log "export $ARM done"
}

rebuild() {
    cd $W/prime-rl
    # --project: export_always_on runs from its own directory, outside prime-rl's project
    PY="uv run --project $W/prime-rl --with gguf python"
    $PY -c "import gguf" > /dev/null
    log "rebuilding R1s from the $ARM weights on GPUs $G0 $G1"
    # routed experts 3-bit (R1), the imatrix-searched 4-bit always-on path (R1s), the rest as the MLX conversion makes it
    B=$W/ornith-$ARM-bf16
    CUDA_VISIBLE_DEVICES=$G0 $PY $SD/repo/01-inference/speed-hillclimb/tensorfold/quantize_experts.py \
        --bf16 $B --base $W/mlx4 --imatrix $SD/Ornith-1.5-35B-A3B-imatrix.gguf \
        --out $W/r1$ARM-experts --gate-up 3 --down 3 > $SD/quantize-experts-$ARM.log 2>&1 &
    (cd $SD/repo/05-compression/ornith/kl && CUDA_VISIBLE_DEVICES=$G1 $PY export_always_on.py --bf16 $B \
        --imatrix $SD/Ornith-1.5-35B-A3B-imatrix.gguf --parts attention linear_attention shared_expert --bits 4 \
        --out $W/always-on-q4-$ARM.safetensors > $SD/always-on-$ARM.log 2>&1) &
    CUDA_VISIBLE_DEVICES= $PY $SD/repo/04-post-training/self-distill/requant_rest.py --bf16 $B --base $W/mlx4 \
        --out $W/rest-$ARM.safetensors > $SD/rest-$ARM.log 2>&1
    wait
    $PY $SD/repo/05-compression/ornith/kl/make_build.py --base $W/r1$ARM-experts \
        --replace $W/always-on-q4-$ARM.safetensors $W/rest-$ARM.safetensors --out $W/r1s-$ARM > $SD/make-build-$ARM.log 2>&1
    log "rebuild $ARM done: $(du -sh $W/r1s-$ARM | cut -f1)"
}

case $STAGE in
    setup) setup ;;
    train) train "$@" ;;
    export) export_weights ;;
    rebuild) rebuild ;;
    all) setup; train "$@"; export_weights; rebuild ;;
    *) echo "unknown stage $STAGE"; exit 1 ;;
esac
