#!/usr/bin/env bash
# pod_engine_runs.sh TAG BIN MODELS...: runs on the pod; tf-qwen35-cuda step/window/tree and beside/resume, compared to base's
TAG=$1; BIN=$2; shift 2
. ~/env.sh
cd ~/runs
IDS="$(cat ids64)"
for m in "$@"; do
  M=$HOME/models/$m
  for mode in step window tree; do $BIN $M $mode "$IDS" $TAG-$m-$mode.bin > $TAG-$m-$mode.log 2>&1; done
  TF_Q35_BESIDE=64 $BIN $M step "$IDS" $TAG-$m-beside.bin > $TAG-$m-beside.log 2>&1
  TF_Q35_PREFILL=32 $BIN $M step "$IDS" $TAG-$m-pre.bin > $TAG-$m-pre.log 2>&1
  TF_Q35_PREFILL=32 TF_Q35_RESUME=1 $BIN $M step "$IDS" $TAG-$m-resume.bin > $TAG-$m-resume.log 2>&1
  for f in step window tree beside pre resume; do cmp -s base-$m-$f.bin $TAG-$m-$f.bin && echo "$TAG $m $f == base" || echo "$TAG $m $f DIFFERS from base"; done
  for f in window tree beside; do cmp -s $TAG-$m-step.bin $TAG-$m-$f.bin && echo "$TAG $m $f == step" || echo "$TAG $m $f != step"; done
  cmp -s $TAG-$m-pre.bin $TAG-$m-resume.bin && echo "$TAG $m resume == fresh" || echo "$TAG $m resume != fresh"
  grep -h "a step" $TAG-$m-step.log
done
echo RUNS_DONE
