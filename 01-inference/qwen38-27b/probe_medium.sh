#!/bin/bash
# The 20-problem math probe on TensorFold, EXL3 4.00 bpw, drafts on, reasoning_effort=medium: the chat template then adds
# no system message, the same prompt the ExLlamaV3 probe builds by hand. Compare with tf-exl3-4.00 (template default
# xhigh, 15/20) and exl3-4.00bpw (ExLlamaV3, 18/20).
cd $(dirname $0)
export TENSORFOLD_KV_BITS=4 TENSORFOLD_ONE_BUFFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TF_DRAFT_PACKED=1 TF_BUDGET_GIB=200
MODEL=/home/riels/.cache/huggingface/hub/models--turboderp--Qwen3.8-27B-exl3/snapshots/113cf7ab958054860e43fb7f3063b1af19171095 \
  DEADLINE_MIN=30 PROBE_ARGS='--template-kwargs {"reasoning_effort":"medium"}' timeout 2400 ./probe_tf.sh tf-exl3-4.00-medium --context 65536
