"""Where the decode matmuls lose bandwidth: each 27B weight shape timed three ways at decode rows.

  qmm      the engine's call (``_mm`` / ``_mm_many``) as forward.py makes it (group sums given, gate+up together)
  read     a plain kernel that only reads the same bytes (16-byte loads, enough blocks to fill the card): the ceiling
           for one launch of that size, launch and ramp included
  graph    the qmm call replayed 20 times inside one CUDA graph (no launch gaps), per call

Usage (from envs/tensorfold): uv run python ../../qwen38-27b/qmm_probe.py [--rows 8] [--targets 192,384]
"""

from __future__ import annotations

import argparse
import collections
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

READ_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void read_kernel(const uint4* __restrict__ p, long long n, unsigned* __restrict__ sink) {
    unsigned acc = 0;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        uint4 v = __ldcs(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) sink[0] = acc;
}
void read_bytes(const at::Tensor& t, const at::Tensor& sink, int blocks) {
    long long n = t.numel() * t.element_size() / 16;
    read_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint4*>(t.data_ptr()), n, reinterpret_cast<unsigned*>(sink.data_ptr()));
}
"""


def snapshot(repo: str) -> Path:
    base = Path.home() / ".cache/huggingface/hub" / ("models--" + repo.replace("/", "--")) / "snapshots"
    return sorted(base.iterdir())[-1]


def timed(fn, reps: int) -> float:
    for _ in range(3):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / reps


def graphed(fn, n: int = 20, reps: int = 20) -> float:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    return timed(g.replay, reps) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--targets", default="192")
    a = ap.parse_args()
    for lock in (Path.home() / ".cache/torch_extensions").glob("*/tensorfold_*/lock"):
        lock.unlink()

    ext = load_inline(name="qmm_probe_read", cpp_sources="void read_bytes(const at::Tensor&, const at::Tensor&, int);",
                      cuda_sources=READ_SRC, functions=["read_bytes"], extra_cuda_cflags=["-O3"])
    sink = torch.zeros(1, dtype=torch.int32, device="cuda")

    from tensorfold.cuda.kernels import qmm
    from tensorfold.families.qwen3_5.cuda.weights import load
    from tensorfold.families.qwen3_5.cuda.forward import _mm, _mm_many

    w = load(snapshot("Vontra/Qwen3.8-27B-MLX-4bit"), tiled=True)
    shapes = collections.OrderedDict()
    for layer in w.layers:
        if layer.linear:
            g = layer.gdn
            calls = [("gdn in", [g.qkv, g.zba] if g.zba is not None else [g.qkv, g.z, g.b, g.a], True), ("gdn out", [g.out], False)]
        else:
            t = layer.attn
            calls = [("attn in", [t.q, t.kv] if t.kv is not None else [t.q, t.k, t.v], True), ("attn o", [t.o], False)]
        calls += [("mlp g+u", [layer.gate, layer.up], True), ("mlp down", [layer.down], False)]
        for kind, ws, many in calls:
            shapes.setdefault((kind, tuple((q.n, q.k) for q in ws)), [ws, many, 0])[2] += 1
    shapes[("head", ((w.head.n, w.head.k),))] = [[w.head], False, 1]

    orig = qmm.split_k
    m = a.rows
    for target in [int(t) for t in a.targets.split(",")]:
        qmm.split_k = lambda n, k, gs=64, target_=None, t=target: orig(n, k, gs, t)
        print(f"\n== {m} rows, split_k target {target}")
        print(f"{'call':10s} {'shape':28s} {'n':>3s} {'sk':>8s} {'MB':>6s} {'qmm us':>8s} {'graph us':>9s} "
              f"{'read us':>8s} {'qmm GB/s':>9s} {'graph':>6s} {'read':>6s}")
        tot = collections.Counter()
        for (kind, dims), (ws, many, count) in shapes.items():
            x = torch.randn(m, ws[0].k, device="cuda", dtype=torch.bfloat16)
            xs = qmm.group_sums(x, ws[0].gs)                  # the model's glue kernels hand these over
            fn = (lambda x=x, ws=ws, xs=xs: _mm_many(x, ws, xs)) if many else (lambda x=x, q=ws[0], xs=xs: _mm(x, q, xs))
            nbytes = sum(q.nbytes() for q in ws)
            us = timed(fn, a.reps)
            gus = graphed(fn)
            flat = torch.cat([t.reshape(-1).view(torch.uint8) for q in ws for t in (q.weight, q.scales, q.biases)])
            rus = min(timed(lambda b=b: ext.read_bytes(flat, sink, b), a.reps) for b in (82 * 4, 82 * 8, 82 * 16))
            sks = ",".join(str(qmm.split_k(q.n, q.k, q.gs)) for q in ws)
            print(f"{kind:10s} {str(dims):28s} {count:3d} {sks:>8s} {nbytes / 1e6:6.1f} {us:8.1f} {gus:9.1f} {rus:8.1f} "
                  f"{nbytes / us / 1e3:9.0f} {nbytes / gus / 1e3:6.0f} {nbytes / rus / 1e3:6.0f}")
            tot["qmm"] += us * count
            tot["graph"] += gus * count
            tot["read"] += rus * count
            del flat
        print(f"round: qmm {tot['qmm'] / 1e3:.2f} ms, graphed {tot['graph'] / 1e3:.2f} ms, "
              f"plain read of the same bytes {tot['read'] / 1e3:.2f} ms")
    qmm.split_k = orig


if __name__ == "__main__":
    main()
