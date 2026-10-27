#!/bin/bash
# Simulated quantized KV cache on the BOS agent rows; status lines to ~/cq.log
. ~/exl3/bin/activate; . ~/compat.sh
one() { local t=$1-agentbos-cq${2/,/-}; KL_ROWS=$HOME/rows/kl-agentbos.safetensors python ~/qkit/model_diff.py -ma ~/q/$1 -mb ~/nanbeige -r 40 -l 2048 -cq $2 > ~/results/kl-$t.log 2>&1 \
  && echo "KL_DONE $t $(grep -o "KL divergence (A, B): *[0-9.]*" ~/results/kl-$t.log | awk "{print \$NF}")" || echo "KL_FAIL $t"; }
{ one 3.5-nb 8; one 3.5-nb 4; } & { one 3.5-nb 6; one 3.5-nb 5; } & wait
echo CQ_DONE
