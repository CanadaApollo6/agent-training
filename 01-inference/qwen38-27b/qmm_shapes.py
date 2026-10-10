"""Time every 4-bit matmul call a Qwen3.8-27B verify round makes, on its own, at decode row counts.

Loads the weights as the engine does (tiled), then replays each layer's calls the way forward.py issues them
(GDN: [qkv, zba] in one call, then out; attention: [q, kv], then o; MLP: gate, up, down; the head once) with CUDA
events, and reports us per call, GB/s and the per-round total by call kind. The 3090 reads ~850 GB/s in practice.

Usage (from envs/tensorfold): uv run python ../../qwen38-27b/qmm_shapes.py [--rows 1,12,16] [--reps 50]
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import torch


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
    return start.elapsed_time(end) * 1000 / reps        # us


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,12,16")
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    for lock in (Path.home() / ".cache/torch_extensions").glob("*/tensorfold_*/lock"):
        lock.unlink()

    from tensorfold.families.qwen3_5.cuda.weights import load
    from tensorfold.families.qwen3_5.cuda.forward import _mm, _mm_many

    w = load(snapshot("Vontra/Qwen3.8-27B-MLX-4bit"), tiled=True)
    calls = []                                             # (kind, fn(x) -> None, input width, weight bytes)
    for layer in w.layers:
        if layer.linear:
            g = layer.gdn
            ws = [g.qkv, g.zba] if g.zba is not None else [g.qkv, g.z, g.b, g.a]
            calls.append(("gdn in [qkv,zba]", ws, True))
            calls.append(("gdn out", [g.out], False))
        else:
            t = layer.attn
            ws = [t.q, t.kv] if t.kv is not None else [t.q, t.k, t.v]
            calls.append(("attn in [q,kv]", ws, True))
            calls.append(("attn o", [t.o], False))
        calls += [("mlp gate", [layer.gate], False), ("mlp up", [layer.up], False), ("mlp down", [layer.down], False)]
    calls.append(("head", [w.head], False))

    shapes = collections.OrderedDict()
    for kind, ws, many in calls:
        key = (kind, tuple((q.n, q.k) for q in ws))
        shapes.setdefault(key, [ws, many, 0])[2] += 1

    for m in [int(r) for r in a.rows.split(",")]:
        print(f"\n== {m} rows")
        total_us = total_bytes = 0.0
        by_kind = collections.Counter()
        for (kind, dims), (ws, many, count) in shapes.items():
            x = torch.randn(m, ws[0].k, device="cuda", dtype=torch.bfloat16)
            fn = (lambda x=x, ws=ws: _mm_many(x, ws)) if many else (lambda x=x, q=ws[0]: _mm(x, q))
            us = timed(fn, a.reps)
            nbytes = sum(q.nbytes() for q in ws)
            total_us += us * count
            total_bytes += nbytes * count
            by_kind[kind] += us * count
            print(f"{kind:18s} {str(dims):34s} x{count:3d}  {us:8.1f} us  {nbytes / us / 1e3:6.0f} GB/s")
        print(f"round: {total_us / 1000:.2f} ms for {total_bytes / 1e9:.2f} GB = {total_bytes / total_us / 1e3:.0f} GB/s "
              f"(at 850 GB/s: {total_bytes / 850e3 / 1000:.2f} ms)")
        for kind, us in by_kind.most_common():
            print(f"  {kind:18s} {us / 1000:6.2f} ms")


if __name__ == "__main__":
    main()
