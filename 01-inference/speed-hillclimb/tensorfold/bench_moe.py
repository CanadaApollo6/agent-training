"""Time one whole MoE block at Ornith 35B's sizes, as the decode path runs it (moe.run: router, top-k, plan, gate/up,
down, combine), in a CUDA graph.

Random weights; router rows are random, so routes are spread like real ones (about 9 distinct experts per row).
Usage: python bench_moe.py [--rows 1 2 4 6] [--widths 4/4 2/3] [--steps old swap select combine]
(each step adds one change: old = the one-warp expert kernel and 6 launches; swap = weights as the mma's A;
select = top-k and plan in one launch; combine = the slots' sum inside down's launch)
"""

import argparse

import torch

from tensorfold.cuda import experts, moe

from bench_experts import D, E, K, layer


STEPS = {"old": (-1, False, 0), "swap": (0, False, 0), "select": (0, True, 0), "combine": (0, True, 64)}


def block(bits_up, bits_down):
    g = torch.Generator(device="cuda").manual_seed(1)
    router = (torch.randn((E + 1, D), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    return moe.Routed(router, layer(bits_up, bits_down), K)


def time(m, rows, iters=200, reps=10):
    x = (torch.randn((rows, D), generator=torch.Generator(device="cuda").manual_seed(rows), device="cuda")
         * 0.5).to(torch.bfloat16)
    for _ in range(5):
        moe.run(x, m)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            moe.run(x, m)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (iters * reps) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 6])
    ap.add_argument("--widths", nargs="+", default=["4/4", "2/3"])
    ap.add_argument("--steps", nargs="+", default=list(STEPS))
    a = ap.parse_args()
    print(f"{'widths':>8} {'step':>8} " + " ".join(f"{f'rows {r}':>9}" for r in a.rows) + "   (us per MoE block)")
    for wd in a.widths:
        m = block(*map(int, wd.split("/")))
        for step in a.steps:
            experts.SPLIT, experts.SELECT, experts.COMBINE = STEPS[step]
            print(f"{wd:>8} {step:>8} " + " ".join(f"{time(m, r):9.1f}" for r in a.rows), flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
