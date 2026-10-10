"""Check and time the CUDA 4-bit tree attention (attn4.cu) against TensorFold's Triton kernels, no model load.

1. accuracy: the shared-chunk partials against PyTorch with the same arithmetic (fp32 up to summation order);
2. exactness: every row of a drafted window (a chain and a tree, windows straddling chunk boundaries) against the
   same row decoded serially (width 1, its path committed to the cache first): bit-identical outputs required;
3. speed: shared-chunk kernel and the whole attention, CUDA graph timed, against TensorFold's.

Usage (from envs/tensorfold): TENSORFOLD_KV_BITS=4 uv run python ../../qwen38-27b/attn4/attn4_test.py [--contexts 20000,100000]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

H, HK, D = 24, 4, 256
G = H // HK


def load_ext():
    from tensorfold.cuda import build
    here = Path(__file__).resolve().parent
    return build.load("attn4_proto", [str(here / "attn4.cu")], extra_cuda_cflags=["-O3", "-lineinfo"],
                      verbose=os.environ.get("VERBOSE") == "1")


def attention4(ext, q, k_nodes, v_nodes, offs, p, scale):
    """``tree_attention.attention`` with the CUDA kernels: window keys and values go in packed, as committed."""

    from tensorfold.cuda.kernels import attention as ta, kvq
    w, h, d = q.shape
    origin = ta.base(q.device).view(torch.int8)
    qr = kvq.rotate(q)
    r = kvq.width(d)
    kn = torch.empty((w, HK, r), dtype=torch.int8, device=q.device)
    vn = torch.empty_like(kn)
    kvq.store(kn, k_nodes)
    kvq.store(vn, v_nodes)
    po = torch.empty((p.chunks, w, h, d), dtype=torch.float32, device=q.device)
    pm = torch.empty((p.chunks, w, h), dtype=torch.float32, device=q.device)
    pl = torch.empty_like(pm)
    mt = min(3, -(-(w * G) // ta.QUERY_TILE))
    ext.shared(qr, origin, offs, p.streams, p.items, po, pm, pl, w, HK, scale, mt)
    ext.tail(qr, origin, offs, p.streams, p.rows, p.paths, p.depths, kn, vn, po, pm, pl, w, HK, scale,
             1 + -(-ta.MAX_NODES // ta.CHUNK))
    out = torch.empty_like(q)
    ta._merge_head[(w, h, d // 128)](po, pm, pl, out, p.streams, p.rows, w, H=h, D=d, DB=128, num_warps=1)
    return kvq.rotate(out)


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


def cache(P, extra, dev):
    from tensorfold.cuda.kernels import kvq
    r = kvq.width(D)
    k = torch.zeros((P + extra, HK, r), dtype=torch.int8, device=dev)
    v = torch.zeros_like(k)
    kvq.store(k[:P], torch.randn(P, HK, D, device=dev, dtype=torch.bfloat16))
    kvq.store(v[:P], torch.randn(P, HK, D, device=dev, dtype=torch.bfloat16))
    return k, v


def exactness(ext, dev) -> bool:
    from tensorfold.cuda.kernels import attention as ta, kvq
    scale = D ** -0.5
    good = True
    cases = [("chain", list(range(-1, 7))), ("tree", [-1, 0, 0, 1, 1, 2, 3, 5])]
    for P in (5, 509, 1020, 1024, 1530, 4000):
        for name, parents in cases:
            w = len(parents)
            k, v = cache(P, 256, dev)
            q = torch.randn(w, H, D, device=dev, dtype=torch.bfloat16)
            kn = torch.randn(w, HK, D, device=dev, dtype=torch.bfloat16)
            vn = torch.randn(w, HK, D, device=dev, dtype=torch.bfloat16)
            offs = torch.tensor(ta.offsets([(k, v)], dev), dtype=torch.int64, device=dev).view(1, 2)
            drafted = attention4(ext, q, kn, vn, offs, ta.plan([parents], [P], G, dev), scale)
            worst = 0
            for t in range(w):
                path, cur = [], t
                while cur >= 0:
                    path.append(cur)
                    cur = parents[cur]
                path = path[::-1]                        # root .. t
                k2, v2 = k.clone(), v.clone()
                n = len(path) - 1
                if n:
                    kvq.store(k2[P:P + n], kn[path[:-1]].contiguous())
                    kvq.store(v2[P:P + n], vn[path[:-1]].contiguous())
                offs2 = torch.tensor(ta.offsets([(k2, v2)], dev), dtype=torch.int64, device=dev).view(1, 2)
                serial = attention4(ext, q[t:t + 1].contiguous(), kn[t:t + 1].contiguous(),
                                    vn[t:t + 1].contiguous(), offs2, ta.plan([[-1]], [P + n], G, dev), scale)
                if not torch.equal(serial[0], drafted[t]):
                    worst = max(worst, (serial[0].float() - drafted[t].float()).abs().max().item())
                    good = False
            print(f"exact P={P:5d} {name}: {'identical' if worst == 0 else f'DIFFERS (max {worst:.3g})'}",
                  flush=True)
    return good


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="20000,100000")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--skip-exact", action="store_true")
    a = ap.parse_args()
    from tensorfold.cuda.kernels import attention as ta, kvq
    assert kvq.BITS == 4, "run with TENSORFOLD_KV_BITS=4"
    torch.manual_seed(0)
    dev = torch.device("cuda")
    ext = load_ext()
    if not a.skip_exact:
        exactness(ext, dev)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from attn_fast import dequant_ref
    scale = D ** -0.5
    for P in [int(x) for x in a.contexts.split(",")]:
        W = a.rows
        k, v = cache(P, 1024, dev)
        q = torch.randn(W, H, D, device=dev, dtype=torch.bfloat16)
        kn = torch.randn(W, HK, D, device=dev, dtype=torch.bfloat16)
        vn = torch.randn(W, HK, D, device=dev, dtype=torch.bfloat16)
        parents = list(range(-1, W - 1))
        plan = ta.plan([parents], [P], G, dev)
        offs = torch.tensor(ta.offsets([(k, v)], dev), dtype=torch.int64, device=dev).view(1, 2)
        origin = ta.base(dev).view(torch.int8)
        qr = kvq.rotate(q)
        full = P // ta.CHUNK
        po = torch.zeros((plan.chunks, W, H, D), dtype=torch.float32, device=dev)
        pm = torch.zeros((plan.chunks, W, H), dtype=torch.float32, device=dev)
        pl = torch.zeros_like(pm)
        mt = min(3, -(-(W * G) // ta.QUERY_TILE))
        new = lambda: ext.shared(qr, origin, offs, plan.streams, plan.items, po, pm, pl, W, HK, scale, mt)
        a_o, a_m, a_l = torch.zeros_like(po), torch.zeros_like(pm), torch.zeros_like(pl)
        old = lambda: ta._shared[(plan.items.shape[0], HK)](qr, origin, origin, offs, plan.streams, plan.items, a_o,
                                                            a_m, a_l, W, H=H, HK=HK, D=D, G=G, CH=ta.CHUNK,
                                                            SCALE=scale, R=kvq.width(D), QBITS=4, QT=ta.QUERY_TILE,
                                                            num_warps=4, num_stages=1)
        us_new, us_old = timed(new), timed(old)
        # reference for the first 8 full chunks (memory)
        nc = min(full, 8)
        kd = dequant_ref(k[:nc * ta.CHUNK], D).float().view(nc, ta.CHUNK, HK, D).repeat_interleave(G, 2)
        vd = dequant_ref(v[:nc * ta.CHUNK], D).float().view(nc, ta.CHUNK, HK, D).repeat_interleave(G, 2)
        sc = torch.einsum("whd,ckhd->cwhk", qr.float(), kd) * scale
        ref_m = sc.amax(-1)
        pr = torch.exp(sc - ref_m[..., None])
        ref_o = torch.einsum("cwhk,ckhd->cwhd", pr, vd) / pr.sum(-1)[..., None]
        got = po[:nc] / pl[:nc, ..., None]
        err = (got - ref_o).abs().max().item()
        ta.ATTN4 = True
        wired = ta.attention(q, kn, vn, offs, plan, scale=scale)
        ta.ATTN4 = False                         # the Triton kernels from here on
        old_out = ta.attention(q, kn, vn, offs, plan, scale=scale)
        new_out = attention4(ext, q, kn, vn, offs, plan, scale)
        print(f"P={P}: TensorFold's attention (TENSORFOLD_ATTN4=1) equals this script's: "
              f"{torch.equal(wired, new_out)}", flush=True)
        diff = (old_out.float() - new_out.float()).abs().max().item()
        all_new = timed(lambda: attention4(ext, q, kn, vn, offs, plan, scale))
        all_old = timed(lambda: ta.attention(q, kn, vn, offs, plan, scale=scale))
        print(f"P={P} rows {W}: shared {us_old:.1f} -> {us_new:.1f} us ({us_old / us_new:.2f}x), partials vs ref "
              f"{err:.2e}; whole attention {all_old:.1f} -> {all_new:.1f} us; |new - old output| {diff:.3g} "
              f"(output scale {old_out.float().abs().max().item():.3g})", flush=True)


if __name__ == "__main__":
    main()
