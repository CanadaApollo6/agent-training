#!/bin/bash
# Reasoning-probe ladder on a pod: for each GGUF, serve it on the pod (setup_pod.sh first) and run
# 01-inference/qwen38-27b/reasoning_probe.py from here through an SSH tunnel.
#
#   05-compression/pod/probe_ladder.sh <ip> <local port> <samples> <file:label> [file:label ...]
#   (files are names in the pod's ~/models; results go to 05-compression/ornith/results)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IP=$1; LPORT=$2; SAMPLES=$3; shift 3
KEY=$HOME/.ssh/prime_ed25519
SLOTS=16
ssh -i $KEY -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -f -N -L "$LPORT:127.0.0.1:8080" "ubuntu@$IP"
for pair in "$@"; do
  FILE=${pair%%:*}; LABEL=${pair#*:}
  echo "== start $LABEL $(date +%H:%M)"
  # 16 slots x (16K answer + prompt)
  ssh -i $KEY "ubuntu@$IP" "~/serve.sh ~/models/$FILE $SLOTS 17408 8080"
  (cd "$ROOT" && PYTHONUNBUFFERED=1 uv run --project 01-inference/envs/prism-llama \
    01-inference/qwen38-27b/reasoning_probe.py server --url "http://127.0.0.1:$LPORT/v1" --label "$LABEL" \
    --slots $SLOTS --samples "$SAMPLES" --out 05-compression/ornith/results > "/tmp/probe-$LABEL.log" 2>&1) \
    && grep "^== $LABEL" "/tmp/probe-$LABEL.log" || { echo "$LABEL failed:"; tail -5 "/tmp/probe-$LABEL.log"; }
done
