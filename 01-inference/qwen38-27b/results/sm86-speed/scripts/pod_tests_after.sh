#!/usr/bin/env bash
# pod_tests_after.sh: runs on the pod after pod_review_fix.sh; zig build test (native, sm_86) on base and both branches
. ~/env.sh
while ! grep -q FIX_DONE ~/runs/fix.log; do sleep 30; done
for d in tf tf-new2 tf-exp2; do
  (cd ~/$d && zig build test native -Dnvcc=/usr/local/cuda-13.0/bin/nvcc -Dsm=86 -Doptimize=ReleaseFast --summary all > ~/runs/fix/ztest-$d.log 2>&1; echo "test exit $?" >> ~/runs/fix/ztest-$d.log)
done
echo TESTS_DONE
