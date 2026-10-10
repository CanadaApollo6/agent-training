"""The segmented decode matmul (weights as the mma's A) against the plain tile form: the same bits at every row count.

Random 4-bit weights in the 27B's shapes, several weights a launch and their own K slices, 1..16 rows; then the time
of each shape at 8 rows.

Usage (from envs/tensorfold): uv run python ../../qwen38-27b/qmm_swap_test.py
"""

from __future__ import annotations

from pathlib import Path

import torch


def rand_q4(qmm, n: int, k: int, gs: int = 64):
    npad = -(-n // 128) * 128
    w = torch.randint(-2 ** 31, 2 ** 31 - 1, (npad // 64, k // gs, 8, 32, gs // 32), dtype=torch.int32, device="cuda")
    s = (torch.rand(k // gs, npad, device="cuda") * 0.02 + 0.001).to(torch.bfloat16)
    b = (torch.randn(k // gs, npad, device="cuda") * 0.05).to(torch.bfloat16)
    return qmm.Q4(w, s, b, n, k, gs)


def main() -> None:
    for lock in (Path.home() / ".cache/torch_extensions").glob("*/tensorfold_*/lock"):
        lock.unlink()
    from tensorfold.cuda.kernels import qmm
    torch.manual_seed(0)
    groups = [[(10240, 5120), (6144, 5120), (48, 5120), (48, 5120)], [(5120, 6144)], [(17408, 5120), (17408, 5120)],
              [(5120, 17408)], [(12288, 5120), (1024, 5120), (1024, 5120)]]
    bad = 0
    for dims in groups:
        qs = [rand_q4(qmm, n, k) for n, k in dims]
        for m in (1, 2, 3, 5, 8, 9, 12, 16):
            x = torch.randn(m, dims[0][1], device="cuda", dtype=torch.bfloat16)
            xs = qmm.group_sums(x)
            for f32 in (False, True):
                qmm.SWAP = True
                a = qmm.matmul_many(x, qs, xs, f32=f32)
                qmm.SWAP = False
                b = qmm.matmul_many(x, qs, xs, f32=f32)
                qmm.SWAP = True
                same = all(torch.equal(u, v) for u, v in zip(a, b))
                bad += not same
                if not same:
                    print(f"DIFFERS {dims} rows {m} f32 {f32}")
    print("swapped == tile form at every shape and row count" if not bad else f"{bad} cases differ")

    for dims in groups:
        qs = [rand_q4(qmm, n, k) for n, k in dims]
        x = torch.randn(8, dims[0][1], device="cuda", dtype=torch.bfloat16)
        xs = qmm.group_sums(x)
        fn = lambda: qmm.matmul_many(x, qs, xs)
        fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(20):
                fn()
        for _ in range(3):
            g.replay()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(20):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        us = e0.elapsed_time(e1) * 1000 / 400
        nbytes = sum(q.nbytes() for q in qs)
        print(f"{str(dims):55s} {us:7.1f} us  {nbytes / us / 1e3:5.0f} GB/s")


if __name__ == "__main__":
    main()
