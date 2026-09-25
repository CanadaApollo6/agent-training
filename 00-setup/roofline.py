"""Measure what this GPU can actually do, and what that means for LLM decode speed.

Decode (one token at a time) reads every weight once per token, so it is bounded by memory
bandwidth: tokens/s <= bandwidth / bytes_of_weights. Prefill and training are big matmuls, so
they are bounded by tensor-core throughput instead. This script measures both ceilings and prints
the decode ceiling for a few models we care about.

    uv run 00-setup/roofline.py
"""
import shutil
import subprocess

import torch

# RTX 3090 datasheet numbers (GA102 whitepaper, reference boost clock 1695 MHz). Factory-overclocked
# cards like the EVGA boost higher, so matmuls land slightly above 100% of these.
SPEC_BANDWIDTH_GBS = 936.0     # 19.5 Gbps GDDR6X x 384-bit bus
SPEC_BF16_TFLOPS = 71.0        # dense tensor cores, FP32 accumulate (what torch uses)
SPEC_TF32_TFLOPS = 35.6

# (name, parameters in billions, effective bits per weight)
MODELS = [
    ("Qwen3.8 27B, bf16 (does not fit)", 27.0, 16.0),
    ("Qwen3.8 27B, ~4.25 bpw (EXL3/Q4)", 27.0, 4.25),
    ("Bonsai 2 27B, ternary PTQ1_0", 27.0, 1.76),
    ("Kev-9B, bf16", 9.0, 16.0),
    ("Kev-4B, bf16", 4.0, 16.0),
    ("Qwen3.5 0.8B, bf16", 0.8, 16.0),
]


def timed(fn, iters):
    """Median-free but warm: run once, then time `iters` runs with CUDA events. Returns seconds/iter."""
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3 / iters


def bandwidth_gbs(nbytes=2 << 30, iters=20):
    """Device-to-device copy: reads nbytes and writes nbytes."""
    src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    return 2 * nbytes / timed(lambda: dst.copy_(src), iters) / 1e9


def matmul_tflops(dtype, n=8192, iters=20):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    return 2 * n**3 / timed(lambda: a @ b, iters) / 1e12


def main():
    assert torch.cuda.is_available(), "CUDA not available: check `nvidia-smi` first"
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), device: {p.name}, sm_{p.major}{p.minor}, {p.multi_processor_count} SMs")
    print(f"memory: {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB, bf16 supported: {torch.cuda.is_bf16_supported()}")
    nvcc = shutil.which("nvcc") or (shutil.which("/opt/cuda/bin/nvcc"))
    nvcc_ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1] if nvcc else "not installed"
    print(f"nvcc: {nvcc_ver}\n")

    bw = bandwidth_gbs()
    torch.backends.cuda.matmul.allow_tf32 = True
    bf16 = matmul_tflops(torch.bfloat16)
    tf32 = matmul_tflops(torch.float32)
    print(f"{'measured':<22}{'achieved':>12}{'spec':>10}{'% of spec':>11}")
    print(f"{'DRAM bandwidth':<22}{bw:>8.0f} GB/s{SPEC_BANDWIDTH_GBS:>6.0f} GB/s{bw / SPEC_BANDWIDTH_GBS:>10.0%}")
    print(f"{'bf16 matmul 8192^3':<22}{bf16:>6.1f} TFLOPS{SPEC_BF16_TFLOPS:>6.1f}{bf16 / SPEC_BF16_TFLOPS:>12.0%}")
    print(f"{'tf32 matmul 8192^3':<22}{tf32:>6.1f} TFLOPS{SPEC_TF32_TFLOPS:>6.1f}{tf32 / SPEC_TF32_TFLOPS:>12.0%}")

    # Ridge point: arithmetic intensity (FLOPs per byte) where a kernel stops being memory-bound.
    print(f"\nridge point: {bf16 * 1e12 / (bw * 1e9):.0f} FLOPs/byte. Decode at batch 1 does ~2 FLOPs per weight,")
    print("i.e. ~1 FLOP/byte in bf16, so it sits far on the memory-bound side.\n")

    print(f"decode ceiling at the measured {bw:.0f} GB/s (weights only; KV cache reads and overheads lower it):")
    print(f"{'model':<36}{'weights':>10}{'tok/s ceiling':>15}")
    for name, params_b, bpw in MODELS:
        gb = params_b * bpw / 8
        fits = "" if gb < total / 1e9 - 1.5 else "  (exceeds VRAM)"
        print(f"{name:<36}{gb:>7.1f} GB{bw / gb:>12.0f}{fits}")


if __name__ == "__main__":
    main()
