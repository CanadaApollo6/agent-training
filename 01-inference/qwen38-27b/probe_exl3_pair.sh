#!/bin/bash
# The 20-problem math probe on TensorFold with turboderp's EXL3 packs, 4.00 then 3.00 bpw (4-bit KV, 64K window).
# Separates weights from engine: compare with tf-kv4-262k / tf-kv16-80k (MLX 4-bit, 15/20) and EXL3-in-ExLlamaV3 (18/20, 19/20).
cd $(dirname $0)
export TENSORFOLD_KV_BITS=4 TENSORFOLD_ONE_BUFFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TF_DRAFT_PACKED=1 TF_BUDGET_GIB=200
MODEL=/home/riels/.cache/huggingface/hub/models--turboderp--Qwen3.8-27B-exl3/snapshots/113cf7ab958054860e43fb7f3063b1af19171095 timeout 2100 ./probe_tf.sh tf-exl3-4.00 --context 65536
MODEL=/home/riels/.cache/huggingface/hub/models--turboderp--Qwen3.8-27B-exl3/snapshots/6fe61ad620abfe97c5b49f9722c2bceeea4ccc28 timeout 2100 ./probe_tf.sh tf-exl3-3.00 --context 65536
