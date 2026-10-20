#!/bin/bash
# pod_build.sh DIR LOG: sync the speed worktree to the pod's DIR and build it there (install, sm_86, ReleaseFast)
set -e
S=$(dirname "$0")
SRC=/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-speed
DIR=${1:-tf-new}
LOG=${2:-build-new.log}
rsync -az --delete --exclude=.git --exclude=zig-out --exclude=.zig-cache -e "ssh -i $HOME/.ssh/prime_ed25519" "$SRC/" ubuntu@129.159.92.102:/home/ubuntu/$DIR/
"$S/pod.sh" "cd ~/$DIR && . ~/env.sh && zig build install -Dnvcc=/usr/local/cuda-13.0/bin/nvcc -Dsm=86 -Doptimize=ReleaseFast > ~/$LOG 2>&1; e=\$?; grep -E 'error|warning: unused' ~/$LOG | grep -v 'diag-suppress' | head -20; echo BUILD_EXIT \$e"
