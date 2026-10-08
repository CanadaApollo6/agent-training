#!/bin/bash
# The 20-problem math probe on TensorFold, EXL3 4.00 bpw, with drafts off ("draft": false per request): does the
# draft model's accept rule cost the points? Compare with tf-exl3-4.00 (drafts on, 15/20) and exl3-4.00bpw (ExLlamaV3, 18/20).
cd $(dirname $0)
export TENSORFOLD_KV_BITS=4 TENSORFOLD_ONE_BUFFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TF_DRAFT_PACKED=1 TF_BUDGET_GIB=200
MODEL=/home/riels/.cache/huggingface/hub/models--turboderp--Qwen3.8-27B-exl3/snapshots/113cf7ab958054860e43fb7f3063b1af19171095 \
  DEADLINE_MIN=70 PROBE_ARGS=--no-draft timeout 4800 ./probe_tf.sh tf-exl3-4.00-nodraft --context 65536
