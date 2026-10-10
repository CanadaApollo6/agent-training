"""Tree attention alone on a synthetic cache: Qwen3.8-27B shapes (24 query heads, 4 kv heads, head_dim 256), a W-row
chain over P committed keys, timed by kernel with CUDA events. No model load: seconds per setting.

Usage (from envs/tensorfold): TENSORFOLD_KV_BITS=4 uv run python ../../qwen38-27b/attn_bench.py [--context 20000] [--rows 8]
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=20000)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    from tensorfold.cuda.kernels import attention as ta, kvq

    torch.manual_seed(0)
    H, HK, D, W, P = 24, 4, 256, a.rows, a.context
    dev = torch.device("cuda")
    r = kvq.width(D)
    k = torch.empty((P + 1024, HK, r), dtype=kvq.dtype(), device=dev)
    v = torch.empty_like(k)
    kvq.store(k[:P], torch.randn(P, HK, D, device=dev, dtype=torch.bfloat16))
    kvq.store(v[:P], torch.randn(P, HK, D, device=dev, dtype=torch.bfloat16))
    q = torch.randn(W, H, D, device=dev, dtype=torch.bfloat16)
    kn = torch.randn(W, HK, D, device=dev, dtype=torch.bfloat16)
    vn = torch.randn(W, HK, D, device=dev, dtype=torch.bfloat16)
    parents = list(range(-1, W - 1))
    plan = ta.plan([parents], [P], H // HK, dev)
    offs = torch.tensor(ta.offsets([(k, v)], dev), dtype=torch.int64, device=dev).view(1, 2)
    f = lambda: ta.attention(q, kn, vn, offs, plan, scale=D ** -0.5)
    out = f()
    for _ in range(5):
        f()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(a.reps):
        f()
    e.record()
    torch.cuda.synchronize()
    us = s.elapsed_time(e) * 1000 / a.reps
    cache = 2 * P * HK * r * k.element_size()
    print(f"rows {W} context {P}: {us:.1f} us a layer (all kernels), cache {cache / 1e6:.1f} MB "
          f"-> {cache / us / 1e3:.0f} GB/s; checksum {out.float().abs().sum().item():.6e}")


if __name__ == "__main__":
    main()
