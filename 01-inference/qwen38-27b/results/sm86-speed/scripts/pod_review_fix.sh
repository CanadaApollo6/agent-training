#!/usr/bin/env bash
# pod_review_fix.sh: runs on the pod; review follow-ups: engine runs (base, qmm, experts) with their == base lines,
# qmm's affine4-decode test and kernel contract, experts' test with times and its five-SM resource usage, fmt and lean
. ~/env.sh
NV=-Dnvcc=/usr/local/cuda-13.0/bin/nvcc
R=~/runs/fix; mkdir -p $R; cd ~/runs
~/scripts/pod_engine_runs.sh base ~/base/tf-qwen35-cuda r1s q27 > $R/engine-base.log 2>&1
~/scripts/pod_engine_runs.sh new ~/tf-new2/zig-out/bin/tf-qwen35-cuda r1s q27 > $R/engine-new.log 2>&1
~/scripts/pod_engine_runs.sh exp ~/tf-exp2/zig-out/bin/tf-qwen35-cuda r1s > $R/engine-exp.log 2>&1
cp ~/runs/*-step.log ~/runs/*-window.log ~/runs/*-tree.log $R/ 2>/dev/null
echo ENGINE_DONE
for d in tf tf-new2 tf-exp2; do (cd ~/$d && zig fmt --check zig build.zig > $R/fmt-$d.txt 2>&1; echo "fmt exit $?" >> $R/fmt-$d.txt; python3 tools/zig/lean_check.py > $R/lean-$d.txt 2>&1; tail -1 $R/lean-$d.txt); done
cd ~/tf-new2
zig-out/bin/tf-cuda-test affine4-decode > $R/a-test.log 2>&1; echo "a-test exit $?" >> $R/a-test.log
zig build kernel-contract $NV -Dsm=86 -Doptimize=ReleaseFast > $R/a-contract.log 2>&1; echo "contract exit $?" >> $R/a-contract.log
cd ~/tf-exp2
zig-out/bin/tf-cuda-test experts3 200 > $R/b-test.log 2>&1; echo "b-test exit $?" >> $R/b-test.log
zig build kernel-contract-fatbins $NV -Dsm=80,86,89,120,121 -Doptimize=ReleaseFast --prefix $HOME/fat-b > $R/b-fat-build.log 2>&1; echo "fat exit $?" >> $R/b-fat-build.log
cuobjdump --dump-resource-usage $HOME/fat-b/fatbin/experts3.fatbin > $R/b-res-usage-all.txt 2>&1
~/tf-exp2/zig-out/bin/tf-cuda-test kernel-contract-limits $HOME/fat-b/fatbin > $R/b-limits.log 2>&1
echo FIX_DONE
