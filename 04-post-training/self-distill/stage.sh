#!/bin/bash
# Copy what pod_train.sh needs to a pod's $W/sd:  ./stage.sh <ssh host, e.g. ubuntu@1.2.3.4> [W=/workspace]
# HOST "local" stages into the local directory W instead (hf_job.sh uploads that to a bucket).
set -euo pipefail
cd "$(dirname "$0")"
HOST=$1; W=${2:-/workspace}
ROOT=$(git rev-parse --show-toplevel)
SSH="ssh -i $HOME/.ssh/prime_ed25519 -o StrictHostKeyChecking=no"
DATA=${DATA:-data/train}   # one or more training sets, e.g. DATA="data/train-a data/train-b"
for d in $DATA; do test -f $d/train.parquet; done
if [ "$HOST" = local ]; then
    SSH="bash -c"; HOST=""; T=""; mkdir -p $W/sd/data $W/sd/repo
else
    T="$HOST:"; $SSH "$HOST" "sudo mkdir -p $W && sudo chown \$(id -u):\$(id -g) $W && mkdir -p $W/sd/data $W/sd/repo"
fi
rsync -a -e "$SSH" pod_train.sh run_arms.sh sft.toml prime-rl-pretokenized.patch finish_export.py requant_rest.py \
    "$HOME/models/ornith/Ornith-1.5-35B-A3B-imatrix.gguf" "$HOME/models/ornith/mlx4/mtp-4bit.safetensors" "$T$W/sd/"
rsync -a -e "$SSH" $DATA "$T$W/sd/data/"
(cd "$ROOT" && rsync -aR -e "$SSH" 01-inference/speed-hillclimb/tensorfold/quantize_experts.py \
    05-compression/ornith/kl/kl_harness.py 05-compression/ornith/kl/export_always_on.py \
    05-compression/ornith/kl/make_build.py "$T$W/sd/repo/")
# requant_rest.py finds quantize_experts.py two levels up from itself, as in the repo
$SSH $HOST "mkdir -p $W/sd/repo/04-post-training/self-distill && cp $W/sd/requant_rest.py $W/sd/repo/04-post-training/self-distill/"
echo "staged to $T$W/sd"
