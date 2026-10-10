"""Prototype: the 4-bit shared-chunk attention kernel without re-interleaving nibbles. A dot product doesn't care about
element order, so low nibbles (even elements) meet the queries' even elements and high nibbles the odd ones; the
output goes back to its places at the store. Compares partials with TensorFold's ``_shared`` and times both.

Usage (from envs/tensorfold): TENSORFOLD_KV_BITS=4 uv run python ../../qwen38-27b/attn_proto.py [--context 20000]
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

from tensorfold.cuda.kernels.attention import _tile
from tensorfold.cuda.kernels.kvq import load_rows


@triton.jit
def _shared2(Q, KC, VC, OFF, STREAM, ITEMS, PO, PM, PL, W, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
             G: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, R: tl.constexpr, QT: tl.constexpr,
             BK: tl.constexpr):
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
        dh = tl.arange(0, D // 2)
        qbase = Q + (node[:, None] * H + head[:, None]) * D
        q_lo = tl.load(qbase + 2 * dh[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
        q_hi = tl.load(qbase + 2 * dh[None, :] + 1, mask=ok[:, None], other=0).to(tl.bfloat16)
        m = tl.full((QT,), float("-inf"), tl.float32)
        l = tl.zeros((QT,), tl.float32)
        o_lo = tl.zeros((QT, D // 2), tl.float32)
        o_hi = tl.zeros((QT, D // 2), tl.float32)
        key = chunk * CH + tl.arange(0, BK)
        for t in range(CH // BK):
            ki = key + t * BK
            krow = koff + (ki * HK + hk) * R
            vrow = voff + (ki * HK + hk) * R
            kraw = tl.load(KC + krow[:, None] + dh[None, :]).to(tl.int32)
            ks = tl.load((KC + krow[:, None] + D // 2).to(tl.pointer_type(tl.float16)) + dh[None, :] // 16).to(tl.float32) * 0.125
            k_lo = (((kraw & 15).to(tl.float32) - 7.5) * ks).to(tl.bfloat16)
            k_hi = ((((kraw >> 4) & 15).to(tl.float32) - 7.5) * ks).to(tl.bfloat16)
            scores = tl.dot(q_lo, tl.trans(k_lo))
            scores = tl.dot(q_hi, tl.trans(k_hi), scores) * SCALE
            tile_m = tl.max(scores, 1)
            next_m = tl.maximum(m, tile_m)
            alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m))
            pr = tl.exp(scores - next_m[:, None])
            vraw = tl.load(VC + vrow[:, None] + dh[None, :]).to(tl.int32)
            vs = tl.load((VC + vrow[:, None] + D // 2).to(tl.pointer_type(tl.float16)) + dh[None, :] // 16).to(tl.float32) * 0.125
            v_lo = (((vraw & 15).to(tl.float32) - 7.5) * vs).to(tl.bfloat16)
            v_hi = ((((vraw >> 4) & 15).to(tl.float32) - 7.5) * vs).to(tl.bfloat16)
            pb = pr.to(tl.bfloat16)
            o_lo = tl.dot(pb, v_lo, o_lo * alpha[:, None])
            o_hi = tl.dot(pb, v_hi, o_hi * alpha[:, None])
            l = l * alpha + tl.sum(pr, 1)
            m = next_m
        base = (chunk * W + node) * H + head
        tl.store(PO + base[:, None] * D + 2 * dh[None, :], o_lo, mask=ok[:, None])
        tl.store(PO + base[:, None] * D + 2 * dh[None, :] + 1, o_hi, mask=ok[:, None])
        tl.store(PM + base, m, mask=ok)
        tl.store(PL + base, l, mask=ok)


@triton.jit
def _shared3(Q, KC, VC, OFF, STREAM, ITEMS, PO, PM, PL, W, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
             G: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, R: tl.constexpr, QT: tl.constexpr,
             BK: tl.constexpr):
    """TensorFold's ``_shared`` with BK keys a tile instead of 64."""

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
        q = tl.load(Q + (node[:, None] * H + head[:, None]) * D + d[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
        m = tl.full((QT,), float("-inf"), tl.float32)
        l = tl.zeros((QT,), tl.float32)
        o = tl.zeros((QT, D), tl.float32)
        for t in range(CH // BK):
            ki = key + t * BK
            every = ki >= 0
            kk = load_rows(KC, koff + (ki * HK + hk) * R, d, every, D, 4)
            vv = load_rows(VC, voff + (ki * HK + hk) * R, d, every, D, 4)
            m, l, o = _tile(q, kk, vv, m, l, o, ki < p, SCALE)
        base = (chunk * W + node) * H + head
        tl.store(PO + base[:, None] * D + d[None, :], o, mask=ok[:, None])
        tl.store(PM + base, m, mask=ok)
        tl.store(PL + base, l, mask=ok)


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
    chunks = plan.chunks
    scale = D ** -0.5

    def bufs():
        return (torch.zeros((chunks, W, H, D), dtype=torch.float32, device=dev),
                torch.zeros((chunks, W, H), dtype=torch.float32, device=dev),
                torch.zeros((chunks, W, H), dtype=torch.float32, device=dev))

    a_o, a_m, a_l = bufs()
    n_items = plan.items.shape[0]
    old = lambda: ta._shared[(n_items, HK)](q, origin, origin, offs, plan.streams, plan.items, a_o, a_m, a_l, W, H=H,
                                            HK=HK, D=D, G=G, CH=ta.CHUNK, SCALE=scale, R=r, QBITS=4,
                                            QT=ta.QUERY_TILE, num_warps=4, num_stages=1)
    us_old = timed(old)
    print(f"_shared (TensorFold): {us_old:.1f} us, {n_items * HK} programs")
    full = P // ta.CHUNK
    kern = {"split": _shared2, "orig": _shared3}
    import os
    which = os.environ.get("KERNEL", "orig")
    for qt in (16, 32):
        items2 = [x for c in range(full) for f in range(0, W * G, qt) for x in (0, f, c)]
        items2 = torch.tensor(items2, dtype=torch.int32, device=dev)
        for bk in (16, 32, 64):
            for warps in (1, 2, 4):
                for stages in (1, 2):
                    b_o, b_m, b_l = bufs()
                    new = lambda: kern[which][(items2.shape[0] // 3, HK)](
                        q, origin, origin, offs, plan.streams, items2, b_o, b_m, b_l, W, H=H, HK=HK, D=D, G=G,
                        CH=ta.CHUNK, SCALE=scale, R=r, QT=qt, BK=bk, num_warps=warps, num_stages=stages)
                    try:
                        us = timed(new)
                    except Exception as exc:
                        print(f"  qt {qt} bk {bk} warps {warps} stages {stages}: failed ({type(exc).__name__})")
                        continue
                    live = a_l[:full] > 0
                    do = (a_o[:full] / a_l[:full, ..., None] - b_o[:full] / b_l[:full, ..., None]).abs()[live].max().item()
                    print(f"  {which} qt {qt} bk {bk} warps {warps} stages {stages}: {us:.1f} us ({us_old / us:.2f}x); "
                          f"max |diff| {do:.1e}", flush=True)

if __name__ == "__main__":
    main()
