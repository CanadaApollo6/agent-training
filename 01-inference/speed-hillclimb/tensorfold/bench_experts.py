"""Time one MoE layer's grouped experts (gate/up then down) at Ornith 35B's sizes, per expert width.

Random weights and routes; rows = tokens verified together (1 for plain decode, depth + 1 with MTP drafts).
Usage: python bench_experts.py
"""

import torch

from tensorfold.cuda import experts

E, D, NI, K, GS = 256, 2048, 512, 8, 64


def tensor(e, n, k, bits, g):
    q = torch.randint(0, 2 ** bits, (e, n, k), generator=g, device="cuda")
    v = (q.to(torch.int64).reshape(e, n, -1, 8) << (bits * torch.arange(8, device="cuda"))).sum(-1)
    b = ((v[..., None] >> (8 * torch.arange(bits, device="cuda"))) & 0xFF).to(torch.uint8)
    words = b.reshape(e, n, -1).contiguous().view(torch.int32)
    s = (torch.rand((e, n, k // GS), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    return words, s, (torch.randn_like(s.float()) * 0.01).to(torch.bfloat16)


def layer(bits_up, bits_down):
    g = torch.Generator(device="cuda").manual_seed(0)
    up = [tensor(E, NI, D, bits_up, g) for _ in range(2)]
    shared = ([tensor(1, NI, D, 4, g) for _ in range(2)], tensor(1, D, NI, 4, g))
    return experts.make(up, tensor(E, D, NI, bits_down, g), GS, shared=shared)


def time(ex, rows, iters=300):
    g = torch.Generator().manual_seed(rows)
    picks = torch.stack([torch.cat([torch.randperm(E, generator=g)[:K], torch.tensor([E])]) for _ in range(rows)])
    picks = picks.to(torch.int32).cuda()
    x = (torch.randn((rows, D), device="cuda") * 0.5).to(torch.bfloat16)
    plan = experts.Plan(rows, K + 1, ex.count, "cuda")
    act = torch.empty((rows * (K + 1), NI), dtype=torch.bfloat16, device="cuda")
    y = torch.empty((rows * (K + 1), D), dtype=torch.float32, device="cuda")

    def step():
        experts.route(picks, plan)
        experts.gate_up(x, ex, plan, act, rows)
        experts.down(act, ex, plan, y, rows)

    for _ in range(20):
        step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            step()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (iters * 10) * 1000  # us per layer


def main():
    print(f"{'widths':>8} " + " ".join(f"{f'rows {r}':>9}" for r in (1, 2, 4, 6)) + "   (us per MoE layer)")
    for bu, bd in ((4, 4), (3, 3), (2, 3), (2, 2)):
        ex = layer(bu, bd)
        print(f"{f'{bu}/{bd}':>8} " + " ".join(f"{time(ex, r):9.1f}" for r in (1, 2, 4, 6)), flush=True)
        del ex
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
