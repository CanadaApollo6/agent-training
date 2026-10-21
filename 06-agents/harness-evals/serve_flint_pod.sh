#!/bin/bash
# On the pod, after setup_flint_pod.sh: Flint BF16 on Recursive's SGLang kit (serve_h200.sh: FA3 attention, Triton MoE,
# fp32 LM head, 262K context, their evaluated memory split), on the pod's 127.0.0.1:30000. Waits until it generates.
# Log: docker logs flint
set -euo pipefail
sudo docker rm -f flint > /dev/null 2>&1 || true
sudo docker run -d --name flint --gpus '"device=0"' --ipc=host -p 127.0.0.1:30000:30000 \
  -v "$HOME/flint:/model" -e MODEL=/model flint-sglang bash /model/sglang/serve_h200.sh "$@" > /dev/null
until curl -sf -m 10 127.0.0.1:30000/health_generate > /dev/null; do
  [ "$(sudo docker inspect -f '{{.State.Running}}' flint)" = true ] || { sudo docker logs --tail 40 flint; exit 1; }
  sleep 15
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo SERVE_DONE
