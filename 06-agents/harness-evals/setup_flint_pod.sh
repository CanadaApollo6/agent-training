#!/bin/bash
# On the pod: download Flint at a pinned revision and build Recursive's SGLang image from its kit. Log: /tmp/setup.log
set -euo pipefail
REV=2a247be0c0aec895ccb0ac7ca92229ec6df37288
command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1; }
export PATH=$HOME/.local/bin:$PATH
sudo docker pull lmsysorg/sglang@sha256:7b6a35df9839fd593a94a1eaee82d7777f472225d9f3ad1f8a2e0cb2bd1785d0 > /tmp/pull.log 2>&1 &
PULL=$!
HF_HUB_ENABLE_HF_TRANSFER=1 uvx --quiet --from 'huggingface_hub[hf_transfer]' hf download RecursiveSI/flint --revision $REV --local-dir ~/flint > /tmp/hf.log 2>&1
echo WEIGHTS_DONE
wait $PULL
cd ~/flint/sglang && sudo docker build -t flint-sglang . > /tmp/build.log 2>&1
echo SETUP_DONE
