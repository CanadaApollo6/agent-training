"""Prototype: a 4-bit cache loader with no number-format conversions, inside TensorFold's shared-chunk attention kernel.

The 3090 runs int -> float and float -> bf16 conversions at 1/8 the rate of float math, and ``kvq.load_rows`` does one
and a half of them per cached value (256 I2FP and 132 F2FP per thread per 64-key tile in ``_shared``'s SASS): more time
than the tile's matrix math. Here a code nibble becomes bf16 by bit placement (0x3F80 | c << 3 is 1 + c/16), two packed
bf16 FMAs give (c - 7.5) / 16 exactly and then times bf16(2 * scale): the value (c - 7.5) * scale / 8 with the scale
rounded to bf16 and one more rounding (different bits from ``load_rows``, so serial and drafted rows must both use it).

Checks the partials against a PyTorch reference of the same arithmetic and times both kernels.

Usage (from envs/tensorfold): TENSORFOLD_KV_BITS=4 uv run python ../../qwen38-27b/attn_fast.py [--context 20000]
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

from tensorfold.cuda.kernels.attention import _tile

# 4 code bytes (8 nibbles, low first) and 4 bf16 scale copies (2 * fp16 scale) -> bf16 values of the low and of the
# high nibbles: [b0 b1] and [b2 b3] spread to 16-bit lanes by prmt, nibble placed as a bf16 mantissa (1 + c/16), minus
# 1.46875 (exact: (c - 7.5)/16), times the scale (one rounding)
_DEQUANT4 = tl.constexpr("""
{
.reg .b32 x, y, t, one, off, nz;
mov.b32 one, 0x3F803F80;
mov.b32 off, 0xBFBCBFBC;
mov.b32 nz, 0x80008000;
prmt.b32 x, $4, 0, 0x4140;
prmt.b32 y, $4, 0, 0x4342;
shl.b32 t, x, 3;
lop3.b32 $0, t, 0x00780078, 0x3F803F80, 0xEA;
shl.b32 t, y, 3;
lop3.b32 $1, t, 0x00780078, 0x3F803F80, 0xEA;
shr.b32 t, x, 1;
lop3.b32 $2, t, 0x00780078, 0x3F803F80, 0xEA;
shr.b32 t, y, 1;
lop3.b32 $3, t, 0x00780078, 0x3F803F80, 0xEA;
fma.rn.bf16x2 $0, $0, one, off;
fma.rn.bf16x2 $1, $1, one, off;
fma.rn.bf16x2 $2, $2, one, off;
fma.rn.bf16x2 $3, $3, one, off;
fma.rn.bf16x2 $0, $0, $5, nz;
fma.rn.bf16x2 $1, $1, $6, nz;
fma.rn.bf16x2 $2, $2, $5, nz;
fma.rn.bf16x2 $3, $3, $6, nz;
}
""")


@triton.jit
def load_rows4(C, row, mask, D: tl.constexpr):
    """bf16 (len(row), D) values of 4-bit cache rows starting at element ``row``, conversion-free."""

    N: tl.constexpr = row.shape[0]
    half = tl.arange(0, D // 2)
    raw = tl.load(C + row[:, None] + half[None, :], mask=mask[:, None], other=0)
    g = tl.arange(0, D // 32)
    scales = tl.load((C + row[:, None] + D // 2).to(tl.pointer_type(tl.float16)) + g[None, :], mask=mask[:, None],
                     other=0.0)
    s2 = (scales.to(tl.float32) * 2.0).to(tl.bfloat16)
    s2 = tl.reshape(tl.broadcast_to(s2[:, :, None], (N, D // 32, 16)), (N, D // 2))
    lo, hi = tl.inline_asm_elementwise(_DEQUANT4, "=r,=r,=r,=r,r,r,r", [raw, s2], (tl.bfloat16, tl.bfloat16),
                                       is_pure=True, pack=4)
    return tl.reshape(tl.join(lo, hi), (N, D))


@triton.jit
def _shared4(Q, KC, VC, OFF, STREAM, ITEMS, PO, PM, PL, W, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
             G: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, R: tl.constexpr, QT: tl.constexpr,
             BK: tl.constexpr = 64):
    """TensorFold's ``_shared`` (4-bit cache) with ``load_rows4`` and BK keys a tile."""

    item = tl.program_id(0)
    hk = tl.program_id(1)
    s = tl.load(ITEMS + item * 3)
    first = tl.load(ITEMS + item * 3 + 1)
    chunk = tl.load(ITEMS + item * 3 + 2)
    start = tl.load(STREAM + s * 4)
    rows = tl.load(STREAM + s * 4 + 1)
    p = tl.load(STREAM + s * 4 + 2)
    if (chunk + 1) * CH <= p:
        koff = tl.load(OFF + s * 2)
        voff = tl.load(OFF + s * 2 + 1)
        rr = first + tl.arange(0, QT)
        ok = rr < rows * G
        node = start + rr // G
        head = hk * G + rr % G
        d = tl.arange(0, D)
        key = chunk * CH + tl.arange(0, BK)
        q = tl.load(Q + (node[:, None] * H + head[:, None]) * D + d[None, :], mask=ok[:, None],
                    other=0).to(tl.bfloat16)
        m = tl.full((QT,), float("-inf"), tl.float32)
        l = tl.zeros((QT,), tl.float32)
        o = tl.zeros((QT, D), tl.float32)
        for t in range(CH // BK):
            ki = key + t * BK
            every = ki >= 0
            kk = load_rows4(KC, koff + (ki * HK + hk) * R, every, D)
            vv = load_rows4(VC, voff + (ki * HK + hk) * R, every, D)
            m, l, o = _tile(q, kk, vv, m, l, o, ki < p, SCALE)
        base = (chunk * W + node) * H + head
        tl.store(PO + base[:, None] * D + d[None, :], o, mask=ok[:, None])
        tl.store(PM + base, m, mask=ok)
        tl.store(PL + base, l, mask=ok)


def dequant_ref(rows: torch.Tensor, d: int) -> torch.Tensor:
    """``load_rows4``'s arithmetic in PyTorch: (n, hk, width) int8 rows -> (n, hk, d) bf16."""

    raw = rows[..., :d // 2].view(torch.uint8).int()
    c = torch.stack([raw & 15, raw >> 4], -1).flatten(-2).float()
    x = ((c - 7.5) / 16).to(torch.bfloat16)                                   # exact
    s2 = (rows[..., d // 2:].contiguous().view(torch.float16).float() * 2).to(torch.bfloat16)
    return (x.unflatten(-1, (d // 32, 32)) * s2[..., None]).flatten(-2)


def timed(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(5):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / (5 * reps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=20000)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--warps", default="4")
    ap.add_argument("--qt", default="16")
    ap.add_argument("--stages", default="1")
    ap.add_argument("--bk", default="64")
    a = ap.parse_args()
    from tensorfold.cuda.kernels import attention as ta, kvq

    torch.manual_seed(0)
    H, HK, D, W, P = 24, 4, 256, a.rows, a.context
    G = H // HK
    dev = torch.device("cuda")
    r = kvq.width(D)
    k = torch.empty((P + 1024, HK, r), dtype=kvq.dtype(), device=dev)
    v = torch.empty_like(k)
    kvq.store(k[:P], torch.randn(P, HK, D, device=dev, dtype=torch.bfloat16))
    kvq.store(v[:P], torch.randn(P, HK, D, device=dev, dtype=torch.bfloat16))
    q = kvq.rotate(torch.randn(W, H, D, device=dev, dtype=torch.bfloat16))
    plan = ta.plan([list(range(-1, W - 1))], [P], G, dev)
    offs = torch.tensor(ta.offsets([(k, v)], dev), dtype=torch.int64, device=dev).view(1, 2)
    origin = ta.base(dev).view(torch.int8)
    chunks, scale, n_items = plan.chunks, D ** -0.5, plan.items.shape[0]
    full = P // ta.CHUNK

    def bufs():
        return (torch.zeros((chunks, W, H, D), dtype=torch.float32, device=dev),
                torch.zeros((chunks, W, H), dtype=torch.float32, device=dev),
                torch.zeros((chunks, W, H), dtype=torch.float32, device=dev))

    a_o, a_m, a_l = bufs()
    old = lambda: ta._shared[(n_items, HK)](q, origin, origin, offs, plan.streams, plan.items, a_o, a_m, a_l, W, H=H,
                                            HK=HK, D=D, G=G, CH=ta.CHUNK, SCALE=scale, R=r, QBITS=4,
                                            QT=ta.QUERY_TILE, num_warps=4, num_stages=1)
    us_old = timed(old)
    print(f"_shared (TensorFold): {us_old:.1f} us, {n_items * HK} programs", flush=True)

    # reference: full chunks of the new arithmetic in PyTorch (fp32), per (row, head)
    kd, vd = dequant_ref(k[:full * ta.CHUNK], D).float(), dequant_ref(v[:full * ta.CHUNK], D).float()
    kh = kd.view(full, ta.CHUNK, HK, D).repeat_interleave(G, 2)            # (chunk, key, head, D)
    vh = vd.view(full, ta.CHUNK, HK, D).repeat_interleave(G, 2)
    sc = torch.einsum("whd,ckhd->cwhk", q.float(), kh) * scale
    ref_m = sc.amax(-1)
    pr = torch.exp(sc - ref_m[..., None])
    ref_o = torch.einsum("cwhk,ckhd->cwhd", pr, vh) / pr.sum(-1)[..., None]

    for qt in [int(x) for x in a.qt.split(",")]:
        items = torch.tensor([x for c in range(full) for f in range(0, W * G, qt) for x in (0, f, c)],
                             dtype=torch.int32, device=dev)
        for warps in [int(x) for x in a.warps.split(",")]:
            for stages, bk in [(st, b) for st in map(int, a.stages.split(",")) for b in map(int, a.bk.split(","))]:
                b_o, b_m, b_l = bufs()
                new = lambda: _shared4[(items.shape[0] // 3, HK)](
                    q, origin, origin, offs, plan.streams, items, b_o, b_m, b_l, W, H=H, HK=HK, D=D, G=G,
                    CH=ta.CHUNK, SCALE=scale, R=r, QT=qt, BK=bk, num_warps=warps, num_stages=stages)
                try:
                    us = timed(new)
                except Exception as exc:
                    print(f"qt {qt} warps {warps} stages {stages}: failed ({type(exc).__name__})", flush=True)
                    continue
                got = b_o[:full] / b_l[:full, ..., None]
                print(f"_shared4 qt {qt} bk {bk} warps {warps} stages {stages}: {us:.1f} us ({us_old / us:.2f}x); "
                      f"max |new - ref| {(got - ref_o).abs().max().item():.2e}", flush=True)

if __name__ == "__main__":
    main()
