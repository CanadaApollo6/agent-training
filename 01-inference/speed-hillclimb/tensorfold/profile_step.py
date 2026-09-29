"""Where a decode step's GPU time goes: TensorFold's Qwen3.5-MoE engine in-process, a greedy generation under the torch
profiler (CUPTI sees the kernels inside CUDA graphs), kernel time grouped by kind.

Usage: python profile_step.py MODEL_DIR [--tokens 256] [--no-drafts] [--top 25]
Sizes its context by GPU memory alone like serve_kernels.py (TF_GPU_ONLY_BUDGET); run it under a memory cap.
"""

import argparse
import collections
import os
import time

import torch

from tensorfold.cuda import capacity

PROMPT = "Write a Python function that parses ISO 8601 durations like P3DT4H5M, with tests."

# kernel name fragments -> kind (first match wins)
KINDS = [
    ("swap_kernel", "experts (grouped, decode)"), ("expert_kernel", "experts (grouped, decode)"),
    ("select_kernel", "routing (top-k + plan)"), ("_router", "routing (router GEMV)"), ("_topk", "routing (top-k)"),
    ("plan", "routing (plan)"), ("combine", "moe combine"),
    ("gdn", "linear attention (GDN)"), ("delta", "linear attention (GDN)"), ("conv", "linear attention (GDN)"),
    ("attn", "attention"), ("flash", "attention"), ("rope", "attention"),
    ("qmm", "quantized linears (qmm)"), ("gemv", "quantized linears (qmm)"), ("_mv", "quantized linears (qmm)"),
    ("norm", "norms"), ("sample", "sampling / verify"), ("argmax", "sampling / verify"), ("verify", "sampling / verify"),
]


def kind(name: str) -> str:
    low = name.lower()
    for frag, k in KINDS:
        if frag.lower() in low:
            return k
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--no-drafts", action="store_true")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--context", type=int, default=8192)
    a = ap.parse_args()

    reserve = float(os.environ.get("TENSORFOLD_CUDA_RESERVE_GIB", "0.3"))
    capacity.available_bytes = lambda t: max(0, int(t.cuda.mem_get_info()[0]) - int(reserve * capacity.GIB))
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine
    from pathlib import Path

    from tokenizers import Tokenizer

    from tensorfold.cuda.markers import TemplateTokens
    from tensorfold.cuda.server import ChatTemplate

    tok = TemplateTokens(Tokenizer.from_file(str(Path(a.model) / "tokenizer.json")), ChatTemplate(Path(a.model)))
    ids = list(tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True,
                                       enable_thinking=False))
    e = Qwen36Engine(a.model, context=a.context, context_explicit=True)
    out = []

    def run():
        out.clear()
        return e.generate(ids, a.tokens, None, lambda t: out.extend(t) and False, draft=not a.no_drafts,
                          stop_eos=False)

    run()                                                        # warm: graphs captured, kernels built
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    stats = run()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    per_name = collections.defaultdict(lambda: [0.0, 0])
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            per_name[ev.name][0] += ev.device_time_total if hasattr(ev, "device_time_total") else ev.cuda_time_total
            per_name[ev.name][1] += 1
    total = sum(v[0] for v in per_name.values())
    by_kind = collections.defaultdict(float)
    for n, (us, _) in per_name.items():
        by_kind[kind(n)] += us
    n = len(out)
    print(f"{n} tokens in {wall:.3f} s unprofiled ({n / wall:.1f} tok/s incl. prefill); stats {stats}")
    print(f"GPU kernel time {total / 1e3:.1f} ms, {total / n:.1f} us a token")
    for k, us in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        print(f"  {k:32s} {us / 1e3:8.2f} ms  {100 * us / total:5.1f}%  {us / n:7.1f} us/token")
    print("top kernels:")
    for name, (us, c) in sorted(per_name.items(), key=lambda kv: -kv[1][0])[:a.top]:
        print(f"  {us / 1e3:8.2f} ms {c:7d}x  {kind(name)[:12]:12s} {name[:110]}")


if __name__ == "__main__":
    main()
