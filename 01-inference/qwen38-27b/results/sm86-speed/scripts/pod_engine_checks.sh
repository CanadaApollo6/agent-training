#!/usr/bin/env bash
# pod_engine_checks.sh TAG BIN: runs on the pod; tf-qwen35-cuda concurrent (beside) and resumed steps for R1s and the 27B
TAG=$1; BIN=$2
. ~/env.sh
cd ~/runs
for m in r1s q27; do
  M=$HOME/models/$m
  TF_Q35_BESIDE=64 $BIN $M step "$(cat ids64)" $TAG-$m-beside.bin > $TAG-$m-beside.log 2>&1
  TF_Q35_PREFILL=32 $BIN $M step "$(cat ids64)" $TAG-$m-pre.bin > $TAG-$m-pre.log 2>&1
  TF_Q35_PREFILL=32 TF_Q35_RESUME=1 $BIN $M step "$(cat ids64)" $TAG-$m-resume.bin > $TAG-$m-resume.log 2>&1
  cmp -s $TAG-$m-step.bin $TAG-$m-beside.bin && echo "$TAG $m beside==step" || echo "$TAG $m beside!=step"
  cmp -s $TAG-$m-pre.bin $TAG-$m-resume.bin && echo "$TAG $m resume==fresh" || echo "$TAG $m resume!=fresh"
done
echo CHECKS_DONE
