"""Lever 3 prototype: R2's MoE experts at rows 1-6, the shared expert merged (current) vs its own 4-bit launch on a
second stream in the graph, the routed launch built without the shared path (TF_NO_SHARED=1 build)."""
import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/Projects/Personal/Research/agent-training/01-inference/speed-hillclimb/tensorfold"))
from bench_experts import tensor, E, D, NI, K, GS
from tensorfold.cuda import experts

g = torch.Generator(device="cuda").manual_seed(0)
up = [tensor(E, NI, D, 2, g) for _ in range(2)]
sh_up = [tensor(1, NI, D, 4, g) for _ in range(2)]
sh_down = tensor(1, D, NI, 4, g)
down = tensor(E, D, NI, 3, g)
merged = experts.make(up, down, GS, shared=(sh_up, sh_down))
routed = experts.make(up, down, GS)
alone = experts.make(sh_up, sh_down, GS)          # the shared expert as a one-expert 4-bit layer


def graph_time(step, iters=300):
    for _ in range(20):
        step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            step()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        graph.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / (iters * 10) * 1000


side = torch.cuda.Stream()
mode = sys.argv[1]
for rows in [int(r) for r in sys.argv[2].split(",")]:
    gg = torch.Generator().manual_seed(rows)
    top = torch.stack([torch.randperm(E, generator=gg)[:K] for _ in range(rows)]).to(torch.int32)
    x = (torch.randn((rows, D), device="cuda") * 0.5).to(torch.bfloat16)
    if mode == "merged":
        picks = torch.cat([top, torch.full((rows, 1), E, dtype=torch.int32)], 1).cuda()
        plan = experts.Plan(rows, K + 1, merged.count, "cuda")
        act = torch.empty((rows * (K + 1), NI), dtype=torch.bfloat16, device="cuda")
        y = torch.empty((rows * (K + 1), D), dtype=torch.float32, device="cuda")

        def step():
            experts.route(picks, plan)
            experts.gate_up(x, merged, plan, act, rows)
            experts.down(act, merged, plan, y, rows)
    else:
        picks = top.cuda()
        plan = experts.Plan(rows, K, routed.count, "cuda")
        act = torch.empty((rows * K, NI), dtype=torch.bfloat16, device="cuda")
        y = torch.empty((rows * K, D), dtype=torch.float32, device="cuda")
        spicks = torch.zeros((rows, 1), dtype=torch.int32, device="cuda")
        splan = experts.Plan(rows, 1, alone.count, "cuda")
        sact = torch.empty((rows, NI), dtype=torch.bfloat16, device="cuda")
        sy = torch.empty((rows, D), dtype=torch.float32, device="cuda")
        experts.route(spicks, splan)               # the shared plan never changes

        def step():
            main = torch.cuda.current_stream()
            experts.route(picks, plan)
            if mode == "stream":
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    experts.gate_up(x, alone, splan, sact, rows)
                    experts.down(sact, alone, splan, sy, rows)
            experts.gate_up(x, routed, plan, act, rows)
            experts.down(act, routed, plan, y, rows)
            if mode == "stream":
                main.wait_stream(side)
            else:                                   # "serial": the shared launches after, one stream
                experts.gate_up(x, alone, splan, sact, rows)
                experts.down(sact, alone, splan, sy, rows)
    print(f"{mode:7s} rows {rows}: {graph_time(step):6.1f} us per layer", flush=True)
