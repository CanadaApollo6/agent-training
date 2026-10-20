#!/bin/bash
# Unpack and build this branch (all artifacts, native included) and the base's native binary; then host tests.
set -u
cd ~
[ -d zig-x86_64-linux-0.17.0 ] || tar xzf zig.tgz
export PATH=$HOME/zig-x86_64-linux-0.17.0:/usr/local/cuda/bin:$PATH
rm -rf tf-kv4 && tar xzf tf-kv4.tgz
NV=/usr/local/cuda/bin/nvcc; $NV --version | tail -2
cd ~/tf-kv4
( time zig build native install -Dnvcc=$NV -Dsm=${SM:-86} -Doptimize=ReleaseFast ) > ~/build-kv4.log 2>&1; echo "kv4 build EXIT $?" | tee -a ~/build-kv4.log
if [ "${BASE:-1}" = 1 ]; then
  cd ~ && rm -rf tf-base && tar xzf tf-base.tgz && cd ~/tf-base
  ( time zig build native -Dnvcc=$NV -Dsm=${SM:-86} -Doptimize=ReleaseFast ) > ~/build-base.log 2>&1; echo "base build EXIT $?" | tee -a ~/build-base.log
fi
cd ~/tf-kv4
[ "${TESTS:-1}" = 1 ] || { echo BUILD_DONE; exit 0; }
( time zig build test-cuda-host ) > ~/test-cuda-host.log 2>&1; echo "test-cuda-host EXIT $?" | tee -a ~/test-cuda-host.log
( time zig build test ) > ~/test.log 2>&1; echo "test EXIT $?" | tee -a ~/test.log
echo BUILD_DONE
