#!/bin/bash
# Exactness of the q4 cache through tf-qwen35-cuda: window, tree, resumed and beside streams equal step bit for bit.
set -u
[ -d /usr/local/cuda/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
B=$HOME/tf-kv4/zig-out/bin/tf-qwen35-cuda
M=$HOME/.cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c
O=$HOME/out/exact; mkdir -p $O
F=${F:-q4}
IDS=$(python3 -c "import json;print(','.join(map(str,json.load(open('$HOME/fixture-8k.json'))['ids'][:${N:-160}])))")
export TF_Q35_PREFILL=${PRE:-1500}
run() { local tag=$1; shift; local t0=$(date +%s.%N); env "$@" >$O/$tag.log 2>&1; local rc=$?; echo "$tag rc=$rc $(python3 -c "print(round($(date +%s.%N)-$t0,1))")s $(grep -E 'steps:' $O/$tag.log)"; }
run step-$F $B $M step $IDS $O/step-$F.bin $F
run window-$F $B $M window $IDS $O/window-$F.bin $F
run tree-$F $B $M tree $IDS $O/tree-$F.bin $F
run resume-$F TF_Q35_RESUME=1 $B $M step $IDS $O/resume-$F.bin $F
run beside-$F TF_Q35_BESIDE=700 $B $M step $IDS $O/beside-$F.bin $F
run prompt-$F $B $M prompt $IDS $O/prompt-$F.bin $F
run presume-$F TF_Q35_RESUME=$((TF_Q35_PREFILL+128)) $B $M prompt $IDS $O/presume-$F.bin $F
for x in window tree resume beside; do cmp -s $O/step-$F.bin $O/$x-$F.bin && echo "$x == step" || echo "$x DIFFERS from step"; done
cmp -s $O/prompt-$F.bin $O/presume-$F.bin && echo "prompt resumed == prompt" || echo "prompt resumed DIFFERS"
ls -la $O/*-$F.bin | awk '{print $5, $9}'
