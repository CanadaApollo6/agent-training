"""In-block K slices (a tile's 2 or 4 slices in one block, summed in shared memory) == the slice buffer, bit for bit.

A launch whose weights all have 2 or 4 K slices takes the in-block form; pairing the same weights with an unsliced one
sends them through the slice buffer instead. Random 4-bit weights in the 27B's and the drafter's shapes, 1..16 rows,
bf16 and f32 outputs.

Usage (from envs/tensorfold): uv run python ../../qwen38-27b/qmm_inblock_test.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from qmm_swap_test import rand_q4


def main() -> None:
    for lock in (Path.home() / ".cache/torch_extensions").glob("*/tensorfold_*/lock"):
        lock.unlink()
    from tensorfold.cuda.kernels import qmm
    torch.manual_seed(0)
    groups = [[(5120, 6144)], [(5120, 17408)], [(5120, 25600)], [(1024, 5120), (1024, 5120)], [(2048, 5120)]]
    bad = cases = 0
    for dims in groups:
        qs = [rand_q4(qmm, n, k) for n, k in dims]
        other = rand_q4(qmm, 256, dims[0][1])          # unsliced: the launch goes through the slice buffer
        for m in (1, 2, 3, 5, 8, 9, 12, 16):
            x = torch.randn(m, dims[0][1], device="cuda", dtype=torch.bfloat16)
            xs = qmm.group_sums(x)
            for f32 in (False, True):
                for sk in (2, 4):
                    a = qmm.matmul_many(x, qs, xs, sks=[sk] * len(qs), f32=f32)
                    b = qmm.matmul_many(x, qs + [other], xs, sks=[sk] * len(qs) + [1], f32=f32)[:len(qs)]
                    cases += 1
                    if not all(torch.equal(u, v) for u, v in zip(a, b)):
                        bad += 1
                        print(f"DIFFERS {dims} rows {m} f32 {f32} slices {sk}")
    print(f"in-block == slice buffer in all {cases} cases" if not bad else f"{bad} of {cases} cases differ")


if __name__ == "__main__":
    main()
