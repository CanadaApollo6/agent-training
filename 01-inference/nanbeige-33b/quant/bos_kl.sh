#!/bin/bash
# KL with BOS restored at position 0, for 4.0 and 3.5-nb; status lines to ~/bos.log
. ~/exl3/bin/activate; . ~/compat.sh
one() { KL_ROWS=$HOME/rows/kl-$2.safetensors python ~/qkit/model_diff.py -ma ~/q/$1 -mb ~/nanbeige -r $3 -l 2048 > ~/results/kl-$1-$2.log 2>&1 \
  && echo "KL_DONE $1-$2 $(grep -o "KL divergence (A, B): *[0-9.]*" ~/results/kl-$1-$2.log | awk "{print \$NF}")" || echo "KL_FAIL $1-$2"; }
for q in 4.0 3.5-nb; do { one $q agentbos 40; one $q wikibos 100; } & done; wait
for q in 3.5 3.0; do { one $q agentbos 40; one $q wikibos 100; } & done; wait
echo BOS_DONE
