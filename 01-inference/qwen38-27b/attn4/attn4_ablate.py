"""Time attn4.cu's shared-chunk kernel with parts removed (ABL 1: no bf16 FMAs, 2: no nibble math either, 3: no
MMAs, 4: no cache loads): which part bounds it. Outputs of ablated builds are meaningless.

Usage (from envs/tensorfold): TENSORFOLD_KV_BITS=4 uv run python ../../qwen38-27b/attn4/attn4_ablate.py [--context 100000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import attn4_test as t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=100000)
    ap.add_argument("--abl", default="0,1,2,3,4")
    ap.add_argument("--tiles", type=int, default=3, help="16-row tiles a block (8 rows: 3)")
    a = ap.parse_args()
    from tensorfold.cuda import build
    from tensorfold.cuda.kernels import attention as ta, kvq
    dev = torch.device("cuda")
    torch.manual_seed(0)
    P, W = a.context, 8
    k, v = t.cache(P, 1024, dev)
    q = kvq.rotate(torch.randn(W, t.H, t.D, device=dev, dtype=torch.bfloat16))
    plan = ta.plan([list(range(-1, W - 1))], [P], t.G, dev)
    offs = torch.tensor(ta.offsets([(k, v)], dev), dtype=torch.int64, device=dev).view(1, 2)
    origin = ta.base(dev).view(torch.int8)
    po = torch.zeros((plan.chunks, W, t.H, t.D), dtype=torch.float32, device=dev)
    pm = torch.zeros((plan.chunks, W, t.H), dtype=torch.float32, device=dev)
    pl = torch.zeros_like(pm)
    src = str(Path(__file__).resolve().parent / "attn4.cu")
    for n in [int(x) for x in a.abl.split(",")]:
        ext = build.load(f"attn4_abl{n}", [src], extra_cuda_cflags=["-O3", f"-DABL={n}"])
        us = t.timed(lambda: ext.shared(q, origin, offs, plan.streams, plan.items, po, pm, pl, W, t.HK, D_SCALE,
                                         a.tiles))
        print(f"ABL {n}: {us:.1f} us", flush=True)


D_SCALE = 256 ** -0.5

if __name__ == "__main__":
    main()
