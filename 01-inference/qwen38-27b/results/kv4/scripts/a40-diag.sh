#!/bin/bash
# a40-diag.sh: after the served run, the 64K fixture with raw keys (q4 values), then raw values (q4 keys).
set -u
until grep -q "A40LONG DONE" ~/a40-long.log 2>/dev/null; do sleep 20; done
export PATH=$HOME/zig-x86_64-linux-0.17.0:/usr/local/cuda/bin:$PATH
M=$HOME/.cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c
[ -d ~/tf-diag ] || cp -a ~/tf-kv4 ~/tf-diag
cd ~/tf-diag
sed -i "s|.q4 => head_dim / 2 + head_dim / 32 \* 2,|.q4 => head_dim / 2 + head_dim / 32 * 2 + 2 * head_dim,|" zig/src/families/qwen3_5/cuda_geometry.zig
for v in KRAW VRAW; do
  { echo "#define $v 1"; cat ~/qwen35_kv4_diag.cu; } > zig/kernels/cuda/qwen35_kv4.cu
  zig build install -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=86 -Doptimize=ReleaseFast > ~/diag-build-$v.log 2>&1 || { echo "build $v failed"; continue; }
  echo "== $v 64k $(date +%T)"
  ./zig-out/bin/tf-qwen35-cuda $M kv-gate ~/fixture-64k.json ~/out/diag64k-$v.json q4 2>&1 | grep -v "^depth.*max [0-9.]*$" | tail -6
done
echo "A40DIAG DONE $(date +%T)"
