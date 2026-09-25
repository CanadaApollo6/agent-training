"""One kernel, many sizes: t(launch) ≈ fixed cost + bytes / bandwidth.

Times a single copy kernel at sizes from 1 KB to 1 GB. The launches are replayed from a CUDA graph, back to back,
so CPU launch overhead is excluded: what remains is the GPU's own per-launch cost plus the data movement.
Each launch copies a different buffer pair, cycling through at least 96 MB, so small copies can't be served from
the 3090's 6 MB L2 cache (in real decode every weight is read once per token, so it never comes from L2 either).

    uv run 01-inference/decode-gap/kernel_size.py
"""
import torch

REPS = 50


def time_per_launch(nbytes):
    pairs = max(1, (96 << 20) // (2 * nbytes))
    bufs = [(torch.empty(nbytes, dtype=torch.uint8, device="cuda"), torch.empty(nbytes, dtype=torch.uint8, device="cuda"))
            for _ in range(pairs)]
    reps = max(REPS, pairs) if nbytes < 64 << 20 else 5
    for dst, src in bufs:
        dst.copy_(src)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            dst, src = bufs[i % pairs]
            dst.copy_(src)
    g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (5 * reps)  # microseconds


def main():
    # Examples of real decode-step launches for Qwen3.5-0.8B (bf16), by the bytes they read.
    examples = {
        2 << 10: "RMSNorm / residual add on one 1,024-wide hidden vector",
        32 << 10: "DeltaNet in_proj_a / in_proj_b weight (16 x 1,024)",
        1 << 20: "k or v projection in a full-attention layer (512 x 1,024)",
        4 << 20: "DeltaNet out_proj or in_proj_z weight (1,024 x 2,048)",
        8 << 20: "one MLP weight matrix (3,584 x 1,024 = 7.3 MB)",
        12 << 20: "DeltaNet in_proj_qkv, already merged in the checkpoint (6,144 x 1,024)",
        64 << 20: "",
        508 << 20: "LM head: the whole 248,320 x 1,024 embedding table",
    }
    print(f"{'bytes read':>12}{'us/launch':>11}{'GB/s':>8}{'fixed-cost share':>18}   example")
    floor = None
    for nbytes, what in examples.items():
        us = time_per_launch(nbytes)
        floor = floor or us
        gbs = 2 * nbytes / (us * 1e-6) / 1e9  # copy reads and writes nbytes
        print(f"{nbytes / 1e6:>9.3f} MB{us:>11.1f}{gbs:>8.0f}{min(floor / us, 1):>18.0%}   {what}")


if __name__ == "__main__":
    main()
