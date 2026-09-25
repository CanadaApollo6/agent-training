"""Where does a decode step's time go? GPU busy time vs wall-clock time, from the PyTorch profiler.

    uv run 01-inference/decode-gap/profile_hf.py --variant fla
"""
import argparse
import sys
import time
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["torch", "fla"], required=True)
parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
parser.add_argument("--tokens", type=int, default=32)
args = parser.parse_args()
if args.variant == "torch":
    sys.modules["fla"] = None  # force the pure-PyTorch DeltaNet path, as in bench_hf.py

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from transformers import AutoTokenizer, Qwen3_5ForCausalLM  # noqa: E402


def main():
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    inputs = tok("Explain how a GPU executes a matrix multiplication.", return_tensors="pt").to("cuda")
    gen = dict(do_sample=False)
    model.generate(**inputs, max_new_tokens=8, min_new_tokens=8, **gen)  # warmup

    # Profile a 1-token run and an N-token run; the difference is N-1 pure decode steps.
    stats = {}
    for n in (1, args.tokens):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            t = time.perf_counter()
            model.generate(**inputs, max_new_tokens=n, min_new_tokens=n, **gen)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
        kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        stats[n] = (wall, sum(e.device_time for e in kernels) / 1e6, len(kernels), Counter(e.name for e in kernels), kernels)

    steps = args.tokens - 1
    wall = (stats[args.tokens][0] - stats[1][0]) / steps * 1e3
    busy = (stats[args.tokens][1] - stats[1][1]) / steps * 1e3
    launches = (stats[args.tokens][2] - stats[1][2]) / steps
    print(f"variant={args.variant}, per decode step:")
    print(f"  wall clock      {wall:6.2f} ms")
    print(f"  GPU busy        {busy:6.2f} ms  ({busy / wall:.0%} of wall; the GPU is idle the rest of the time)")
    print(f"  kernel launches {launches:6.0f}      ({wall * 1e3 / launches:.1f} us of wall time per launch)")
    print("  most frequent kernels per step:")
    per_step = stats[args.tokens][3] - stats[1][3]
    for name, count in per_step.most_common(12):
        print(f"    {count / steps:5.0f}  {name[:100]}")

    # GPU time by kind of kernel, per decode step: what could fusion remove, and what can't it?
    print("\n  GPU time per step by category:")
    print(f"    {'category':<34}{'kernels':>8}{'total us':>10}{'us/kernel':>11}{'share':>8}")
    busy_us = busy * 1e3
    for cat, (n, us) in sorted(categorize(stats, steps).items(), key=lambda kv: -kv[1][1]):
        print(f"    {cat:<34}{n:>8.0f}{us:>10.0f}{us / n:>11.1f}{us / busy_us:>8.0%}")


def category(name):
    n = name.lower()
    if "gemv" in n or "gemm" in n or "cublas" in n or "cutlass" in n:
        return "matmul/matvec (reads weights)"
    if "flash" in n or "fmha" in n or "attention" in n or "sdpa" in n:
        return "attention (full-attn layers)"
    if "memcpy" in n or "memset" in n or "copy" in n:
        return "copies and casts"
    if "reduce" in n:
        return "reductions (mean, sum, argmax)"
    if "elementwise" in n or "index" in n or "cat" in n:
        return "elementwise (add, mul, rsqrt, ...)"
    return "Triton / other (fla, conv)"


def categorize(stats, steps):
    """Per-step kernel count and GPU microseconds for each category (N-token run minus 1-token run)."""
    agg = {}
    for sign, n in ((1, args.tokens), (-1, 1)):
        for e in stats[n][4]:
            c = agg.setdefault(category(e.name), [0.0, 0.0])
            c[0] += sign / steps
            c[1] += sign * e.device_time / steps
    return agg


if __name__ == "__main__":
    main()
