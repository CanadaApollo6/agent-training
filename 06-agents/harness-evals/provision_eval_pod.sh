#!/bin/bash
# Provision a fresh A6000 pod to serve an Ornith build with TensorFold, then serve it on the pod's 127.0.0.1:8080.
#
#   ./provision_eval_pod.sh <pod ip> <build dir under ~/models/ornith, e.g. r1s> <served name, e.g. ornith35b-r1s>
#
# Copies TensorFold, its env lock and serve_kernels.py, runs setup_r1s_pod.sh (CUDA 13 toolkit, g++-12, uv), copies
# the build (rsync -L: R1s's shards are hard links into R1), then starts serve_r1s_pod.sh. Log: /tmp/provision-<ip>.log.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=$1; BUILD=$2; NAME=$3
S="ssh -i $HOME/.ssh/prime_ed25519 -o StrictHostKeyChecking=accept-new"
ssh-keygen -R "$P" > /dev/null 2>&1 || true
until $S -o ConnectTimeout=10 ubuntu@$P true 2> /dev/null; do sleep 10; done
scp -q -i ~/.ssh/prime_ed25519 06-agents/harness-evals/setup_r1s_pod.sh 06-agents/harness-evals/serve_r1s_pod.sh ubuntu@$P:
$S ubuntu@$P 'mkdir -p ~/at/01-inference/tools ~/at/01-inference/envs/tensorfold ~/at/01-inference/speed-hillclimb/tensorfold ~/models/ornith; setsid nohup bash ~/setup_r1s_pod.sh > /tmp/setup.log 2>&1 < /dev/null &'
rsync -a -e "$S" --exclude .git --exclude __pycache__ --exclude .venv 01-inference/tools/TensorFold ubuntu@$P:at/01-inference/tools/
rsync -a -e "$S" 01-inference/envs/tensorfold/pyproject.toml 01-inference/envs/tensorfold/uv.lock ubuntu@$P:at/01-inference/envs/tensorfold/
rsync -a -e "$S" 01-inference/speed-hillclimb/tensorfold/serve_kernels.py ubuntu@$P:at/01-inference/speed-hillclimb/tensorfold/
rsync -aL -e "$S" ~/models/ornith/$BUILD ubuntu@$P:models/ornith/
until $S ubuntu@$P 'grep -q SETUP_DONE /tmp/setup.log'; do sleep 15; done
$S ubuntu@$P "MODEL_DIR=\$HOME/models/ornith/$BUILD NAME=$NAME bash ~/serve_r1s_pod.sh"
